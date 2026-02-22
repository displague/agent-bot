import asyncio
import collections
import logging
import os
import subprocess
import tempfile
import threading
import sys
from typing import Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime dependency may be absent
    np = None
try:
    import soundfile as sf
except Exception:  # pragma: no cover - runtime dependency may be absent
    sf = None

from config import (
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    VOICE_CHUNK_SECONDS,
    VOICE_CONTROL_PREFIX,
    VOICE_INTERJECT_POLICY,
    VOICE_MIN_UTTERANCE_SECONDS,
    VOICE_OFFLINE_INFER_TIMEOUT_SECONDS,
    VOICE_SAMPLE_RATE,
    VOICE_SILENCE_SECONDS,
    VOICE_STOP_GRACE_SECONDS,
    VOICE_VAD_RMS_THRESHOLD,
)
from utils import (
    capture_microphone_chunk,
    play_wav_file_interruptible,
    run_personaplex_offline,
    PersonaPlexStreamingSession,
)

logger = logging.getLogger("autonomous_system.voice_loop")


def parse_control_prefix(text: str, prefix: str = VOICE_CONTROL_PREFIX) -> Tuple[str, str]:
    raw = (text or "")
    if not raw.strip():
        return "S", ""
    if raw.startswith(prefix) and len(raw) >= 2:
        code = raw[1].upper()
        if code in {"S", "P", "I"}:
            return code, raw[2:].lstrip()
        return "S", raw
    return "S", raw.strip()


class VoiceLoop:
    def __init__(self, state, interaction_log_manager, personaplex_manager=None, audio_multiplexer=None):
        self.state = state
        self.interaction_log_manager = interaction_log_manager
        self.personaplex_manager = personaplex_manager
        self.audio_multiplexer = audio_multiplexer
        self._stop_event = asyncio.Event()
        self._listen_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        self._utterance_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: Optional[asyncio.Queue] = None
        self._playback_interrupt = threading.Event()
        self._speaking = False
        self._policy = VOICE_INTERJECT_POLICY.lower()
        self._streaming_session: Optional[PersonaPlexStreamingSession] = None
        self._streaming_audio_buffer = []
        self._streaming_text_buffer = ""

    def get_recent_audio(self, seconds: float = 5.0) -> np.ndarray:
        """Retrieve the last 'seconds' of audio from the rolling buffer."""
        chunks_to_get = int(seconds / VOICE_CHUNK_SECONDS)
        available = list(self._rolling_buffer)
        tail = available[-chunks_to_get:] if chunks_to_get < len(available) else available
        if not tail:
            return np.array([], dtype=np.float32)
        return np.concatenate(tail)

    async def start(self):
        if self.is_running:
            return
        if np is None or sf is None:
            missing = []
            if np is None:
                missing.append("numpy")
            if sf is None:
                missing.append("soundfile")
            raise RuntimeError(
                "voice loop dependencies missing: " + ", ".join(missing)
            )
        self._stop_event.clear()
        if self.audio_multiplexer:
            self._audio_queue = self.audio_multiplexer.subscribe()
        else:
            raise RuntimeError("AudioMultiplexer required for VoiceLoop.")
            
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        self._set_state(mode="offline-continuous", server="running", session="local")
        # Audible cue that voice control is active.
        try:
            if os.name == "nt":
                import winsound  # type: ignore

                winsound.MessageBeep(winsound.MB_OK)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
        await self._set_activity("listening", "voice loop started")

    async def stop(self):
        self._stop_event.set()
        self._playback_interrupt.set()
        tasks = [t for t in (self._listen_task, self._process_task) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=VOICE_STOP_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Voice loop stop timed out after %.1fs; forcing state transition.",
                    VOICE_STOP_GRACE_SECONDS,
                )
        if self.audio_multiplexer and self._audio_queue:
            self.audio_multiplexer.unsubscribe(self._audio_queue)
            self._audio_queue = None
        self._listen_task = None
        self._process_task = None
        self._speaking = False
        self._set_state(mode="offline-disabled", server="stopped", session="disconnected")
        await self._set_activity("idle", "voice loop stopped")

    @property
    def is_running(self) -> bool:
        return (
            self._listen_task is not None
            and not self._listen_task.done()
            and self._process_task is not None
            and not self._process_task.done()
        )

    def _set_state(self, *, mode: Optional[str] = None, server: Optional[str] = None, session: Optional[str] = None):
        if mode is not None:
            self.state["voice_mode"] = mode
        if server is not None:
            self.state["voice_server_state"] = server
        if session is not None:
            self.state["voice_session_state"] = session

    async def _set_activity(self, activity: str, event: Optional[str] = None):
        self.state["voice_activity_state"] = activity
        if event:
            self.state["voice_last_event"] = event
            await self.interaction_log_manager.append(f"Voice: {event}")

    async def _listen_loop(self):
        chunks = []
        silence_s = 0.0
        speaking_s = 0.0
        while not self._stop_event.is_set():
            try:
                if self._audio_queue is None:
                    break
                chunk = await self._audio_queue.get()
                chunk = np.asarray(chunk).reshape(-1)
                if chunk.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
                is_speech = rms >= VOICE_VAD_RMS_THRESHOLD

                # Full-duplex streaming path
                if self.personaplex_manager and not self._speaking:
                    if self._streaming_session is None:
                        self._streaming_session = self.personaplex_manager.create_session(
                            PERSONAPLEX_TEXT_PROMPT, PERSONAPLEX_VOICE_PROMPT
                        )
                        # We run start in a thread to not block the listen loop
                        await asyncio.to_thread(self._streaming_session.start)
                    
                    out_audio, out_text = await asyncio.to_thread(self._streaming_session.step, chunk)
                    
                    if out_text:
                        self._streaming_text_buffer += out_text
                        # Periodically update the UI with what the model is transcribing/thinking
                        if len(self._streaming_text_buffer) % 20 == 0:
                            await self._set_activity("thinking", f"Model: {self._streaming_text_buffer[-40:]}")

                    if out_audio is not None and np.abs(out_audio).max() > 0.01:
                        # Model is starting to speak!
                        # In true full-duplex, we'd stream this audio out. 
                        # For now, we collect it and we'll play it when the user stops or if model is confident.
                        self._streaming_audio_buffer.append(out_audio)

                if is_speech and self._speaking and self._policy == "hard":
                    self._playback_interrupt.set()
                    await self._set_activity("interrupted", "hard interruption requested")

                if is_speech:
                    chunks.append(chunk)
                    speaking_s += VOICE_CHUNK_SECONDS
                    silence_s = 0.0
                elif chunks:
                    chunks.append(chunk)
                    silence_s += VOICE_CHUNK_SECONDS

                if chunks and silence_s >= VOICE_SILENCE_SECONDS:
                    if speaking_s >= VOICE_MIN_UTTERANCE_SECONDS:
                        if self._streaming_session and self._streaming_audio_buffer:
                            # In streaming mode, we already have the output audio!
                            combined_audio = np.concatenate(self._streaming_audio_buffer)
                            # Put it in a format process_loop expects or create a new path
                            # For simplicity now, we'll put it in the queue with a special key
                            await self._utterance_queue.put({
                                "type": "streaming_result",
                                "audio": combined_audio,
                                "text": self._streaming_text_buffer
                            })
                            self._streaming_audio_buffer = []
                            self._streaming_text_buffer = ""
                            # Reset session for next turn
                            self._streaming_session = None
                        else:
                            utterance = np.concatenate(chunks)
                            await self._utterance_queue.put(utterance)
                    chunks = []
                    silence_s = 0.0
                    speaking_s = 0.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                err_msg = str(exc) or type(exc).__name__
                logger.error("Voice listen loop error: %s", err_msg)
                await self._set_activity("error", f"listen error: {err_msg[:80]}")
                await asyncio.sleep(0.2)

    async def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._utterance_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            try:
                if isinstance(item, dict) and item.get("type") == "streaming_result":
                    # Streaming result path: we already have audio and text
                    audio_data = item["audio"]
                    text = item["text"]
                    
                    control, spoken_text = parse_control_prefix(text)
                    await self.interaction_log_manager.append(
                        f"Voice model (streaming) [{control}]: {spoken_text or '<empty>'}"
                    )
                    
                    if control in {"S", "I"}:
                        # Save to temp file for playback
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            tmp_output = f.name
                        sf.write(tmp_output, audio_data, VOICE_SAMPLE_RATE)
                        
                        try:
                            self._speaking = True
                            self._playback_interrupt.clear()
                            await self._set_activity(
                                "speaking",
                                "interjecting" if control == "I" else "speaking",
                            )
                            interrupted = await asyncio.to_thread(
                                play_wav_file_interruptible,
                                tmp_output,
                                self._playback_interrupt,
                            )
                            self._speaking = False
                            if interrupted:
                                await self._set_activity("interrupted", "playback interrupted")
                            else:
                                await self._set_activity("listening", "playback complete")
                        finally:
                            if os.path.exists(tmp_output):
                                os.unlink(tmp_output)
                    else:
                        await self._set_activity("passive", "model chose passive mode")
                        await self._set_activity("listening", "resuming listen mode")
                    continue

                # Legacy utterance path
                utterance = item
                await self._set_activity("thinking", "processing utterance")
                with tempfile.TemporaryDirectory(prefix="voice_loop_") as tmpdir:
                    input_wav = os.path.join(tmpdir, "input.wav")
                    output_wav = os.path.join(tmpdir, "output.wav")
                    output_text = os.path.join(tmpdir, "output.json")
                    sf.write(input_wav, utterance, VOICE_SAMPLE_RATE)
                    
                    if self.personaplex_manager:
                        await self.personaplex_manager.infer_async(
                            text_prompt=PERSONAPLEX_TEXT_PROMPT,
                            voice_prompt_path=PERSONAPLEX_VOICE_PROMPT,
                            input_wav_path=input_wav,
                            output_wav_path=output_wav,
                            output_text_path=output_text
                        )
                        # Text tokens need manual decoding since PersonaPlexManager currently focus on audio
                        # We still use the offline helper for text for now or extend manager
                        from utils import _decode_output_tokens
                        text = _decode_output_tokens(output_text)
                    else:
                        result = await run_personaplex_offline(
                            input_wav,
                            output_wav,
                            output_text=output_text,
                            voice_prompt=PERSONAPLEX_VOICE_PROMPT,
                            text_prompt=PERSONAPLEX_TEXT_PROMPT,
                            timeout_seconds=VOICE_OFFLINE_INFER_TIMEOUT_SECONDS,
                        )
                        text = result.get("generated_text", "")
                    
                    control, spoken_text = parse_control_prefix(text)
                    await self.interaction_log_manager.append(
                        f"Voice model [{control}]: {spoken_text or '<empty>'}"
                    )
                    if control in {"S", "I"}:
                        self._speaking = True
                        self._playback_interrupt.clear()
                        await self._set_activity(
                            "speaking",
                            "interjecting" if control == "I" else "speaking",
                        )
                        interrupted = await asyncio.to_thread(
                            play_wav_file_interruptible,
                            output_wav,
                            self._playback_interrupt,
                        )
                        self._speaking = False
                        if interrupted:
                            await self._set_activity("interrupted", "playback interrupted")
                        else:
                            await self._set_activity("listening", "playback complete")
                    else:
                        await self._set_activity("passive", "model chose passive mode")
                        await self._set_activity("listening", "resuming listen mode")
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                self._speaking = False
                logger.error("Voice process timed out.")
                await self._set_activity("error", "process timeout (model too slow)")
            except Exception as exc:
                self._speaking = False
                logger.error("Voice process loop error: %s", exc)
                await self._set_activity("error", f"process error: {str(exc)[:80]}")
