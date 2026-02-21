import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

logger = logging.getLogger("autonomous_system.utils")


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


def play_wav_file(path: str) -> None:
    """Play a WAV file through the default output device."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot play audio.")
    data, sample_rate = sf.read(path, always_2d=False)
    sd.play(data, samplerate=sample_rate)
    sd.wait()


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

    candidates = [
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


def run_personaplex_offline(
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

    voice_name = Path(voice_prompt).name
    resolved_voice_dir = voice_prompt_dir or str(Path(voice_prompt).parent)

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

    logger.info("Running PersonaPlex offline inference.")
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
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Set it in environment/.env before starting voice server.")

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
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
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


def transcribe_audio(audio_waveform, sample_rate: int = VOICE_SAMPLE_RATE) -> str:
    """Compatibility wrapper: convert waveform via PersonaPlex and return generated text."""
    with tempfile.TemporaryDirectory(prefix="agentbot_voice_") as tmpdir:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output.wav")
        output_text = os.path.join(tmpdir, "output.json")
        sf.write(input_wav, np.asarray(audio_waveform), sample_rate)
        response = run_personaplex_offline(
            input_wav,
            output_wav,
            output_text=output_text,
        )
        return response.get("generated_text", "")
