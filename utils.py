import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import numpy as np
import soundfile as sf

from config import (
    PERSONAPLEX_SERVER_LOG_PATH,
    PERSONAPLEX_CPU_OFFLOAD,
    PERSONAPLEX_DEVICE,
    PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS,
    PERSONAPLEX_PYTHON_BIN,
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    PERSONAPLEX_VOICE_PROMPT_DIR,
    VOICE_SAMPLE_RATE,
)

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - dependency may be optional at runtime
    sd = None
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency may be optional at runtime
    load_dotenv = None

try:
    from moshi.offline import run_inference as moshi_run_inference
    from moshi.models import MimiModel, LMGen, loaders
    import sentencepiece
except ImportError:
    moshi_run_inference = None
    MimiModel = None
    LMGen = None
    loaders = None
    sentencepiece = None

logger = logging.getLogger("autonomous_system.utils")


class PersonaPlexManager:
    """Manages persistent in-process PersonaPlex models for fast inference."""
    
    def __init__(self, device: str = PERSONAPLEX_DEVICE, cpu_offload: bool = PERSONAPLEX_CPU_OFFLOAD, status_callback: Optional[Callable[[str], None]] = None):
        self.device = device
        self.cpu_offload = cpu_offload
        self.status_callback = status_callback
        self.mimi = None
        self.other_mimi = None
        self.lm = None
        self.text_tokenizer = None
        self.repo = "nvidia/personaplex-7b-v1"
        self._lock = threading.Lock()

    def _status(self, message: str):
        logger.info(message)
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def load(self):
        """Load models into VRAM if not already loaded."""
        if self.mimi is not None:
            return
        
        if loaders is None:
            logger.error("moshi package not available for in-process loading.")
            raise RuntimeError("moshi package not available for in-process loading.")

        self._status("PersonaPlexManager: starting in-process model load...")
        with self._lock:
            from huggingface_hub import hf_hub_download
            
            try:
                # 1) Load Mimi
                self._status("PersonaPlexManager: loading mimi...")
                mimi_weight = hf_hub_download(self.repo, loaders.MIMI_NAME)
                self.mimi = loaders.get_mimi(mimi_weight, self.device)
                self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
                
                # 2) Load tokenizer
                self._status("PersonaPlexManager: loading tokenizer...")
                tokenizer_path = hf_hub_download(self.repo, loaders.TEXT_TOKENIZER_NAME)
                self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
                
                # 3) Load Moshi LM
                self._status("PersonaPlexManager: loading moshi lm...")
                moshi_weight = hf_hub_download(self.repo, loaders.MOSHI_NAME)
                self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=self.cpu_offload)
                self.lm.eval()
                
                # Streaming forever setup
                self.mimi.streaming_forever(1)
                self.other_mimi.streaming_forever(1)
                self._status("PersonaPlexManager: models loaded and ready.")
            except Exception as e:
                logger.exception("PersonaPlexManager: failed to load models: %s", e)
                raise

    async def infer_async(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Run a single inference turn asynchronously."""
        return await asyncio.to_thread(self.infer, text_prompt, voice_prompt_path, input_wav_path, output_wav_path, output_text_path)

    def infer(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Synchronous inference implementation using the warm models."""
        logger.info("PersonaPlexManager: starting inference for prompt: %s", text_prompt[:100])
        self.load()
        from moshi.offline import warmup, wrap_with_system_tags
        
        # Use provided prompt or default
        final_text_prompt = text_prompt or PERSONAPLEX_TEXT_PROMPT
        
        # Resolve voice prompt path
        v_path = Path(voice_prompt_path)
        v_name = v_path.name
        v_dir = PERSONAPLEX_VOICE_PROMPT_DIR.strip()
        if not v_dir:
            v_parent = str(v_path.parent)
            if v_parent not in {"", "."}:
                v_dir = v_parent
        
        final_voice_prompt = str(Path(v_dir) / v_name) if v_dir else v_name
        logger.debug("PersonaPlexManager: using voice prompt path: %s", final_voice_prompt)

        with self._lock:
            # Re-create LMGen for each turn to reset state correctly for now
            lm_gen = LMGen(
                self.lm,
                audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                sample_rate=self.mimi.sample_rate,
                device=self.device,
                frame_rate=self.mimi.frame_rate,
                save_voice_prompt_embeddings=False,
                use_sampling=True,
                temp=0.8, # More natural for fillers
                temp_text=0.7,
                top_k=1,
                top_k_text=1,
            )
            lm_gen.streaming_forever(1)
            
            frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            warmup(self.mimi, self.other_mimi, lm_gen, self.device, frame_size)
            
            # Load prompts
            if final_voice_prompt.endswith('.pt'):
                lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
            else:
                lm_gen.load_voice_prompt(final_voice_prompt)
            
            lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(final_text_prompt))
            
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(self.mimi)
            
            # Process input WAV
            import soundfile as sf
            input_data, _ = sf.read(input_wav_path, dtype="float32")
            if input_data.ndim > 1:
                input_data = input_data.mean(-1)
            
            logger.info("PersonaPlexManager: processing input audio (%d samples)...", len(input_data))
            all_out_pcms = []
            all_text_tokens = []
            chunk_size = frame_size
            for i in range(0, len(input_data), chunk_size):
                chunk = input_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                chunk_ts = torch.from_numpy(chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                codes = self.mimi.encode(chunk_ts)
                
                # Single step
                tokens = lm_gen.step(codes)
                
                # Capture text tokens (k=0)
                text_token = tokens[0, 0].item()
                if text_token not in {0, 1, 2, 3}: # Skip BOS/EOS/PAD
                    all_text_tokens.append(self.text_tokenizer.IdToPiece(text_token))

                # Decode agent tokens (k=1..dep_q)
                out_pcm = self.other_mimi.decode(tokens[:, 1:])
                all_out_pcms.append(out_pcm.cpu().detach().numpy().squeeze())
            
            if all_out_pcms:
                res = np.concatenate(all_out_pcms)
                sf.write(output_wav_path, res, self.mimi.sample_rate)
                logger.info("PersonaPlexManager: saved generated audio to %s", output_wav_path)
            
            if output_text_path:
                with open(output_text_path, "w", encoding="utf-8") as f:
                    json.dump(all_text_tokens, f)
                logger.debug("PersonaPlexManager: saved generated text tokens to %s", output_text_path)
            
            return output_wav_path


class PersonaPlexServerHandle:
    def __init__(self, process: subprocess.Popen, ssl_dir: str, log_path: str, log_file):
        self.process = process
        self.ssl_dir = ssl_dir
        self.log_path = log_path
        self.log_file = log_file


def extract_text_features(text: str) -> Dict[str, Any]:
    """Basic text features used for lightweight routing/debugging."""
    cleaned = text.strip()
    return {
        "length": len(cleaned),
        "word_count": len(cleaned.split()) if cleaned else 0,
    }


def extract_audio_features(audio_waveform: np.ndarray) -> Dict[str, Any]:
    """Basic audio features used for monitoring captured audio quality."""
    if audio_waveform is None:
        return {}
    waveform = np.asarray(audio_waveform)
    if waveform.size == 0:
        return {"samples": 0}
    return {
        "samples": int(waveform.size),
        "rms": float(np.sqrt(np.mean(np.square(waveform.astype(np.float32))))),
        "peak_abs": float(np.max(np.abs(waveform))),
    }


def capture_microphone_to_wav(
    output_wav: str,
    duration_seconds: int,
    sample_rate: int = VOICE_SAMPLE_RATE,
) -> str:
    """Record microphone input and save it as mono WAV."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot capture microphone audio.")
    frames = int(duration_seconds * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(output_wav, recording, sample_rate)
    return output_wav


def capture_microphone_chunk(
    duration_seconds: float,
    sample_rate: int = VOICE_SAMPLE_RATE,
) -> np.ndarray:
    """Capture a short mono chunk from the default microphone."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot capture microphone audio.")
    frames = max(1, int(duration_seconds * sample_rate))
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return np.asarray(recording).reshape(-1)


def play_wav_file(path: str) -> None:
    """Play a WAV file through the default output device."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot play audio.")
    data, sample_rate = sf.read(path, always_2d=False)
    sd.play(data, samplerate=sample_rate)
    sd.wait()


def play_wav_file_interruptible(path: str, stop_event: Optional[threading.Event] = None) -> bool:
    """Play a WAV file and optionally stop early when stop_event is set."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot play audio.")
    data, sample_rate = sf.read(path, always_2d=False)
    sd.play(data, samplerate=sample_rate)
    interrupted = False
    while True:
        stream = sd.get_stream()
        if stream is None or not stream.active:
            break
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            sd.stop()
            break
        time.sleep(0.02)
    return interrupted


def _decode_output_tokens(output_text_path: str) -> str:
    if not os.path.exists(output_text_path):
        return ""
    with open(output_text_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return ""
    filtered = [t for t in payload if t not in {"EPAD", "BOS", "EOS", "PAD"}]
    return "".join(filtered).strip()


def _resolve_personaplex_python() -> str:
    if PERSONAPLEX_PYTHON_BIN:
        return PERSONAPLEX_PYTHON_BIN

    # Prioritize the .venv relative to project root
    candidates = [
        Path(sys.prefix) / "Scripts" / "python.exe",
        Path(sys.prefix) / "bin" / "python",
        Path(".venv/Scripts/python.exe"),
        Path(".venv/bin/python"),
        Path("personaplex/.venv/Scripts/python.exe"),
        Path("personaplex/.venv/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _load_env_if_present() -> None:
    if load_dotenv is None:
        return
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


async def run_personaplex_offline(
    input_wav: str,
    output_wav: str,
    *,
    output_text: Optional[str] = None,
    voice_prompt: str = PERSONAPLEX_VOICE_PROMPT,
    text_prompt: str = PERSONAPLEX_TEXT_PROMPT,
    voice_prompt_dir: str = PERSONAPLEX_VOICE_PROMPT_DIR,
    device: str = PERSONAPLEX_DEVICE,
    cpu_offload: bool = PERSONAPLEX_CPU_OFFLOAD,
    timeout_seconds: int = PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS,
    seed: int = 42424242,
) -> Dict[str, Any]:
    """Run PersonaPlex offline inference through moshi.offline CLI."""
    _load_env_if_present()
    if output_text is None:
        output_text = os.path.join(tempfile.gettempdir(), "personaplex_output_text.json")

    voice_path = Path(voice_prompt)
    voice_name = voice_path.name
    resolved_voice_dir = voice_prompt_dir.strip() if isinstance(voice_prompt_dir, str) else ""
    if not resolved_voice_dir:
        parent = str(voice_path.parent)
        # If only a basename is provided (e.g. NATF2.pt), let moshi.offline
        # resolve/download voices instead of forcing current directory.
        if parent not in {"", "."}:
            resolved_voice_dir = parent

    command = [
        _resolve_personaplex_python(),
        "-m",
        "moshi.offline",
        "--input-wav",
        input_wav,
        "--output-wav",
        output_wav,
        "--output-text",
        output_text,
        "--voice-prompt",
        voice_name,
        "--text-prompt",
        text_prompt,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if resolved_voice_dir:
        command.extend(["--voice-prompt-dir", resolved_voice_dir])
    if cpu_offload:
        command.append("--cpu-offload")

    # Attempt direct in-process inference if available
    if moshi_run_inference is not None:
        logger.info("Running PersonaPlex offline inference directly (in-process).")
        try:
            await asyncio.to_thread(
                moshi_run_inference,
                input_wav=input_wav,
                output_wav=output_wav,
                output_text=output_text,
                text_prompt=text_prompt,
                voice_prompt_path=str(Path(resolved_voice_dir) / voice_name) if resolved_voice_dir else voice_name,
                tokenizer_path=None,
                moshi_weight=None,
                mimi_weight=None,
                hf_repo="nvidia/personaplex-7b-v1",
                device=device,
                seed=seed,
                temp_audio=0.0, # Default greedy
                temp_text=0.0,
                topk_audio=1,
                topk_text=1,
                greedy=True,
                save_voice_prompt_embeddings=False,
                cpu_offload=cpu_offload
            )
            generated_text = _decode_output_tokens(output_text)
            return {
                "output_wav": output_wav,
                "output_text_path": output_text,
                "generated_text": generated_text,
                "stdout": "in-process execution",
            }
        except Exception as e:
            logger.warning("Direct PersonaPlex inference failed: %s. Falling back to subprocess.", e)

    logger.info("Running PersonaPlex offline inference via subprocess.")
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        logger.error("PersonaPlex offline inference failed: %s", result.stderr)
        raise RuntimeError(
            "PersonaPlex offline inference failed. "
            f"Exit code: {result.returncode}. Stderr: {result.stderr.strip()}"
        )

    generated_text = _decode_output_tokens(output_text)
    return {
        "output_wav": output_wav,
        "output_text_path": output_text,
        "generated_text": generated_text,
        "stdout": result.stdout,
    }


def start_personaplex_server(cpu_offload: bool = PERSONAPLEX_CPU_OFFLOAD) -> PersonaPlexServerHandle:
    """Start PersonaPlex server mode for full-duplex interaction."""
    _load_env_if_present()

    ssl_dir = tempfile.mkdtemp(prefix="personaplex_ssl_")
    os.makedirs(os.path.dirname(PERSONAPLEX_SERVER_LOG_PATH), exist_ok=True)
    log_file = open(PERSONAPLEX_SERVER_LOG_PATH, "a", encoding="utf-8")
    log_file.write("\n=== Starting PersonaPlex server ===\n")
    log_file.flush()
    command = [
        _resolve_personaplex_python(),
        "-m",
        "moshi.server",
        "--ssl",
        ssl_dir,
    ]
    if cpu_offload:
        command.append("--cpu-offload")
    logger.info("Starting PersonaPlex server mode.")
    child_env = os.environ.copy()
    # Prefer offline cache when token is absent; avoids unnecessary hub calls.
    if not child_env.get("HF_TOKEN"):
        child_env.setdefault("HF_HUB_OFFLINE", "1")
        child_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=child_env,
    )
    time.sleep(1.5)
    if process.poll() is not None:
        try:
            with open(PERSONAPLEX_SERVER_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-20:]).strip()
        except Exception:
            tail = ""
        log_file.close()
        shutil.rmtree(ssl_dir, ignore_errors=True)
        detail = f" Exit code: {process.returncode}."
        if tail:
            detail += f" Server log tail: {tail}"
        raise RuntimeError("PersonaPlex server exited during startup." + detail)

    return PersonaPlexServerHandle(
        process=process,
        ssl_dir=ssl_dir,
        log_path=PERSONAPLEX_SERVER_LOG_PATH,
        log_file=log_file,
    )


def stop_personaplex_server(handle: Optional[PersonaPlexServerHandle]) -> None:
    """Stop a running PersonaPlex server process and clean temporary SSL files."""
    if handle is None:
        return
    try:
        if handle.process.poll() is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=5)
    finally:
        if getattr(handle, "log_file", None):
            try:
                handle.log_file.close()
            except Exception:
                pass
        shutil.rmtree(handle.ssl_dir, ignore_errors=True)


async def transcribe_audio(audio_waveform, sample_rate: int = VOICE_SAMPLE_RATE) -> str:
    """Compatibility wrapper: convert waveform via PersonaPlex and return generated text."""
    with tempfile.TemporaryDirectory(prefix="agentbot_voice_") as tmpdir:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output.wav")
        output_text = os.path.join(tmpdir, "output.json")
        sf.write(input_wav, np.asarray(audio_waveform), sample_rate)
        response = await run_personaplex_offline(
            input_wav,
            output_wav,
            output_text=output_text,
        )
        return response.get("generated_text", "")
