import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import collections
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from mss import mss
from PIL import Image

from config import (
    PERSONAPLEX_SERVER_LOG_PATH,
    PERSONAPLEX_CPU_OFFLOAD,
    PERSONAPLEX_DEVICE,
    PERSONAPLEX_USE_CUDA_GRAPHS,
    PERSONAPLEX_OPTIMIZE,
    PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS,
    PERSONAPLEX_PYTHON_BIN,
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    PERSONAPLEX_VOICE_PROMPT_DIR,
    VOICE_SAMPLE_RATE,
    VOICE_CHUNK_SECONDS,
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


def _ensure_voice_prompt_exists(voice_name: str, repo_id: str = "nvidia/personaplex-7b-v1") -> str:
    """Ensure the voice prompt exists locally, downloading from HF if needed."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import tarfile

    # If it's an absolute path and exists, use it
    v_path = Path(voice_name)
    if v_path.is_absolute() and v_path.exists():
        return str(v_path)

    # Check common local locations
    project_root = Path(__file__).parent.absolute()
    v_dir = PERSONAPLEX_VOICE_PROMPT_DIR.strip()
    
    local_search_dirs = [Path("."), project_root / "voices", project_root / "personaplex"]
    if v_dir:
        local_search_dirs.append(Path(v_dir))
        
    for search_dir in local_search_dirs:
        if not search_dir.exists():
            continue
        # Recursive search for the filename
        for root, _, files in os.walk(search_dir):
            if voice_name in files:
                found_path = str(Path(root) / voice_name)
                logger.debug("Found voice prompt locally at: %s", found_path)
                return found_path

    # Not found locally, attempt HF download
    logger.info("Voice prompt '%s' not found locally. Attempting HF download...", voice_name)
    try:
        files = list_repo_files(repo_id=repo_id)
        if voice_name in files:
            return hf_hub_download(repo_id=repo_id, filename=voice_name)
        if f"voices/{voice_name}" in files:
            return hf_hub_download(repo_id=repo_id, filename=f"voices/{voice_name}")
        
        # Fallback to voices.tgz
        if "voices.tgz" in files:
            logger.info("Prompt not found as standalone. Downloading and extracting voices.tgz...")
            tgz_path = hf_hub_download(repo_id=repo_id, filename="voices.tgz")
            project_root = Path(__file__).parent.absolute()
            extract_dir = project_root / "voices"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            
            # Find it in extracted dir
            for root, _, files in os.walk(extract_dir):
                if voice_name in files:
                    return str(Path(root) / voice_name)
    except Exception as e:
        logger.error("Failed to download voice prompt from HF: %s", e)
    
    return voice_name


class PersonaPlexStreamingSession:
    """Manages an active, stateful streaming session with PersonaPlex models."""
    
    def __init__(self, manager, text_prompt: str, voice_prompt_path: str):
        self.manager = manager
        self.text_prompt = text_prompt
        self.voice_prompt_path = voice_prompt_path
        self.lm_gen = None
        self.frame_size = 0
        self._lock = threading.Lock()
        self.all_text_tokens = []
        self._is_ready = False

    def start(self):
        """Initialize the session: load models, run warmup, and step system prompts."""
        from moshi.offline import warmup, wrap_with_system_tags
        import moshi.models.lm
        
        # Patch CUDAGraphed based on manager's optimization strategy
        self.manager._apply_optimizations()
        
        self.manager.load()
        
        # Resolve voice prompt path
        final_voice_prompt = _ensure_voice_prompt_exists(self.voice_prompt_path)

        with self._lock:
            self.lm_gen = LMGen(
                self.manager.lm,
                audio_silence_frame_cnt=int(0.5 * self.manager.mimi.frame_rate),
                sample_rate=self.manager.mimi.sample_rate,
                device=self.manager.device,
                frame_rate=self.manager.mimi.frame_rate,
                save_voice_prompt_embeddings=False,
                use_sampling=True,
                temp=0.8,
                temp_text=0.7,
                top_k=1,
                top_k_text=1,
            )
            self.lm_gen.streaming_forever(1)
            self.frame_size = int(self.manager.mimi.sample_rate / self.manager.mimi.frame_rate)
            
            if PERSONAPLEX_USE_CUDA_GRAPHS:
                self.manager._status("PersonaPlexManager: warming up CUDA graphs...")
                warmup(self.manager.mimi, self.manager.other_mimi, self.lm_gen, self.manager.device, self.frame_size)
            
            if final_voice_prompt.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
            else:
                self.lm_gen.load_voice_prompt(final_voice_prompt)
            
            self.lm_gen.text_prompt_tokens = self.manager.text_tokenizer.encode(wrap_with_system_tags(self.text_prompt))
            
            self.manager.mimi.reset_streaming()
            self.manager.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.lm_gen.step_system_prompts(self.manager.mimi)
            self._is_ready = True

    def step(self, audio_chunk: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Process a single audio chunk and return generated audio and text (if any).."""
        if not self._is_ready:
            self.start()
            
        with self._lock:
            all_out_pcms = []
            new_texts = []
            
            # Process the chunk in frame_size increments
            for i in range(0, len(audio_chunk), self.frame_size):
                sub_chunk = audio_chunk[i : i + self.frame_size]
                if len(sub_chunk) < self.frame_size:
                    sub_chunk = np.pad(sub_chunk, (0, self.frame_size - len(sub_chunk)))
                
                # Ensure contiguous for torch.from_numpy
                sub_chunk = np.ascontiguousarray(sub_chunk)
                chunk_ts = torch.from_numpy(sub_chunk).to(self.manager.device).unsqueeze(0).unsqueeze(0)
                codes = self.manager.mimi.encode(chunk_ts)
                
                # Model step
                tokens = self.lm_gen.step(codes)
                
                # Decode text token
                text_token = tokens[0, 0].item()
                if text_token not in {0, 1, 2, 3}:
                    piece = self.manager.text_tokenizer.IdToPiece(text_token)
                    self.all_text_tokens.append(piece)
                    new_texts.append(piece)
                
                # Decode audio tokens (k=1..dep_q+1)
                out_pcm = self.manager.other_mimi.decode(tokens[:, 1 : self.manager.lm.dep_q + 1])
                all_out_pcms.append(out_pcm.cpu().detach().numpy().squeeze())
            
            final_audio = np.concatenate(all_out_pcms) if all_out_pcms else None
            final_text = "".join(new_texts) if new_texts else None
            
            return final_audio, final_text


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
        # Dedicated executor for sequential, low-jitter inference
        from concurrent.futures import ThreadPoolExecutor
        self.step_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="personaplex_step")

    def shutdown(self):
        """Shut down the manager and its executors."""
        if hasattr(self, "step_executor"):
            self.step_executor.shutdown(wait=True)
            logger.info("PersonaPlexManager: step_executor shut down.")

    def _status(self, message: str):
        logger.info(message)
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def _apply_optimizations(self):
        """Apply torch.compile and CUDA graph optimizations based on config."""
        # Fix Moshi's broken streaming propagation logic
        try:
            import moshi.modules.streaming
            def patched_apply_named_streaming(self, fn):
                def _handle_module(prefix: str, module: torch.nn.Module, recurse: bool = True, is_root: bool = False):
                    propagate = True
                    if isinstance(module, moshi.modules.streaming.StreamingModule):
                        # The BUG: Moshi skips the module if _streaming_propagate is False,
                        # even if it's the root we just called .streaming() on!
                        if module._streaming_propagate or is_root:
                            fn(prefix, module)
                        else:
                            propagate = False
                    if not recurse:
                        return
                    if propagate:
                        for name, child in module.named_children():
                            _handle_module(prefix + ("." if prefix else "") + name, child)

                # Pass is_root=True for the initial call
                _handle_module("", self, recurse=False, is_root=True)
                for name, child in self.named_children():
                    _handle_module(name, child)
            
            moshi.modules.streaming.StreamingModule._apply_named_streaming = patched_apply_named_streaming
            logger.info("PersonaPlexManager: patched StreamingModule._apply_named_streaming to fix propagation bug.")
        except Exception as e:
            logger.warning("PersonaPlexManager: failed to patch moshi StreamingModule: %s", e)

        opt = PERSONAPLEX_OPTIMIZE.lower()
        if opt == "eager":
            logger.info("PersonaPlexManager: forcing eager mode (no optimizations).")
            self._patch_cuda_graphs(disable=True)
            os.environ["NO_TORCH_COMPILE"] = "1"
            return

        # Handle 'auto' or explicit 'compile'/'graphs'
        use_compile = (opt in ["auto", "compile", "graphs"])
        use_graphs = (opt in ["auto", "graphs"])

        if use_compile:
            logger.info("PersonaPlexManager: enabling torch.compile (lazy).")
            os.environ.pop("NO_TORCH_COMPILE", None)
        else:
            os.environ["NO_TORCH_COMPILE"] = "1"

        if use_graphs:
            logger.info("PersonaPlexManager: allowing CUDA graphs.")
            self._patch_cuda_graphs(disable=False)
        else:
            self._patch_cuda_graphs(disable=True)

    def _patch_cuda_graphs(self, disable: bool):
        """Patch moshi.models.lm.CUDAGraphed to set the disable flag."""
        self._last_patch_disable = disable
        try:
            import moshi.models.lm
            original_init = moshi.models.lm.CUDAGraphed.__init__
            def patched_init(self, func, warmup_steps=1, disable_orig=False, **kwargs):
                # We ignore the model's 'disable' hint and use our global one
                # Note: We must support 'disable' as a keyword if it's passed that way
                target_disable = kwargs.get('disable', disable_orig)
                # But we override it with our global preference
                original_init(self, func, warmup_steps=warmup_steps, disable=disable)
            moshi.models.lm.CUDAGraphed.__init__ = patched_init
            logger.info("PersonaPlexManager: patched moshi CUDAGraphed (disable=%s)", disable)
        except Exception as e:
            logger.warning("PersonaPlexManager: failed to patch moshi CUDA graphs: %s", e)

    def load(self):
        """Load models into VRAM if not already loaded."""
        if self.mimi is not None:
            return
        
        if loaders is None:
            logger.error("moshi package not available for in-process loading.")
            raise RuntimeError("moshi package not available for in-process loading.")

        # Apply optimizations before loading modules
        self._apply_optimizations()

        def get_vram():
            if torch is not None and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            return 0.0

        vram_before = get_vram()
        self._status("PersonaPlexManager: starting in-process model load...")
        with self._lock:
            from huggingface_hub import hf_hub_download
            
            try:
                # 1) Load Mimi
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading mimi...")
                mimi_weight = hf_hub_download(self.repo, loaders.MIMI_NAME)
                self.mimi = loaders.get_mimi(mimi_weight, self.device)
                self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: mimi loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                vram_before = vram_now
                
                # 2) Load tokenizer
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading tokenizer...")
                tokenizer_path = hf_hub_download(self.repo, loaders.TEXT_TOKENIZER_NAME)
                self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: tokenizer loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                vram_before = vram_now
                
                # 3) Load Moshi LM
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading moshi lm...")
                moshi_weight = hf_hub_download(self.repo, loaders.MOSHI_NAME)
                # Ensure device is correctly passed and cpu_offload is handled
                self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=self.cpu_offload)
                self.lm.eval()
                if not self.cpu_offload:
                    self.lm.to(self.device)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: moshi loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                
                # Streaming forever setup
                self.mimi.streaming_forever(1)
                self.other_mimi.streaming_forever(1)
                self._status("PersonaPlexManager: models loaded and ready.")
            except Exception as e:
                logger.exception("PersonaPlexManager: failed to load models: %s", e)
                raise

    def create_session(self, text_prompt: str, voice_prompt_path: str) -> PersonaPlexStreamingSession:
        """Create a new streaming session."""
        return PersonaPlexStreamingSession(self, text_prompt, voice_prompt_path)

    def get_status(self) -> Dict[str, Any]:
        """Return detailed status of the manager and models."""
        with self._lock:
            return {
                "loaded": self.mimi is not None,
                "device": self.device,
                "cpu_offload": self.cpu_offload,
                "optimize_strategy": PERSONAPLEX_OPTIMIZE,
                "torch_compile": os.environ.get("NO_TORCH_COMPILE") != "1",
                "cuda_graphs": getattr(self, "_last_patch_disable", None) is False,
            }

    async def infer_async(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Run a single inference turn asynchronously."""
        return await asyncio.to_thread(self.infer, text_prompt, voice_prompt_path, input_wav_path, output_wav_path, output_text_path)

    def infer(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Synchronous inference implementation using the warm models."""
        logger.info("PersonaPlexManager: starting inference for prompt: %s", text_prompt[:100])
        import moshi.models.lm
        
        # Apply optimizations based on strategy
        self._apply_optimizations()
        
        self.load()
        from moshi.offline import warmup, wrap_with_system_tags
        
        # Use provided prompt or default
        final_text_prompt = text_prompt or PERSONAPLEX_TEXT_PROMPT
        
        # Resolve voice prompt path
        final_voice_prompt = _ensure_voice_prompt_exists(voice_prompt_path)
            
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
            if PERSONAPLEX_USE_CUDA_GRAPHS:
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
                
                # Ensure contiguous for torch.from_numpy
                chunk = np.ascontiguousarray(chunk)
                chunk_ts = torch.from_numpy(chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                codes = self.mimi.encode(chunk_ts)
                
                # Single step
                tokens = lm_gen.step(codes)
                
                # Capture text tokens (k=0)
                text_token = tokens[0, 0].item()
                if text_token not in {0, 1, 2, 3}: # Skip BOS/EOS/PAD
                    piece = self.text_tokenizer.IdToPiece(text_token)
                    all_text_tokens.append(piece)

                # Decode agent tokens (k=1..dep_q+1)
                # dep_q is the number of audio codebooks
                out_pcm = self.other_mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
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


class AudioMultiplexer:
    """Captures microphone audio and broadcasts chunks to multiple subscribers."""
    
    def __init__(self, sample_rate: int = VOICE_SAMPLE_RATE, chunk_seconds: float = VOICE_CHUNK_SECONDS):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.subscribers: Set[Tuple[asyncio.Queue, Optional[asyncio.AbstractEventLoop]]] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        """Create a new queue and add it to subscribers."""
        queue = asyncio.Queue()
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        with self._lock:
            self.subscribers.add((queue, loop))
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a queue from subscribers."""
        with self._lock:
            to_remove = [s for s in self.subscribers if s[0] == queue]
            for s in to_remove:
                self.subscribers.remove(s)

    def start(self):
        """Start the background capture thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_capture(self):
        """Dedicated thread for sounddevice capture to ensure steady timing."""
        logger.info("AudioMultiplexer: starting capture thread.")
        try:
            while not self._stop_event.is_set():
                chunk = capture_microphone_chunk(self.chunk_seconds, self.sample_rate)
                if chunk is not None and chunk.size > 0:
                    with self._lock:
                        for q, loop in list(self.subscribers):
                            try:
                                if loop:
                                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                                else:
                                    q.put_nowait(chunk)
                            except asyncio.QueueFull:
                                pass
                            except Exception:
                                pass
        except Exception as e:
            logger.error("AudioMultiplexer: capture thread error: %s", e)
        finally:
            logger.info("AudioMultiplexer: capture thread stopped.")


class RollingAudioBuffer:
    """Independent auditory memory that feeds from AudioMultiplexer."""
    
    def __init__(self, multiplexer: AudioMultiplexer, seconds: float = 10.0):
        self.multiplexer = multiplexer
        self.seconds = seconds
        max_chunks = int(seconds / multiplexer.chunk_seconds)
        self.buffer = collections.deque(maxlen=max_chunks)
        self.queue = multiplexer.subscribe()
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the async task to pull chunks from the multiplexer queue."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the buffer task and unsubscribe."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.multiplexer.unsubscribe(self.queue)

    async def _run_loop(self):
        try:
            while not self._stop_event.is_set():
                chunk = await self.queue.get()
                self.buffer.append(chunk)
        except asyncio.CancelledError:
            pass

    def get_recent(self, seconds: float = 5.0) -> np.ndarray:
        """Retrieve the last 'seconds' of audio from the buffer."""
        chunks_to_get = int(seconds / self.multiplexer.chunk_seconds)
        available = list(self.buffer)
        tail = available[-chunks_to_get:] if chunks_to_get < len(available) else available
        if not tail:
            return np.array([], dtype=np.float32)
        return np.concatenate(tail)


class PersonaPlexServerHandle:
    def __init__(self, process: subprocess.Popen, ssl_dir: str, log_path: str, log_file):
        self.process = process
        self.ssl_dir = ssl_dir
        self.log_path = log_path
        self.log_file = log_file


def capture_screen() -> Image.Image:
    """Capture the primary monitor and return as a PIL Image."""
    with mss() as sct:
        # Get the primary monitor
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL Image
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


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

    # Resolve voice prompt path
    final_voice_path_str = _ensure_voice_prompt_exists(voice_prompt)

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
        final_voice_path_str,
        "--text-prompt",
        text_prompt,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
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
                    voice_prompt_path=final_voice_path_str,
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
