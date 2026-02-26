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
import utils

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
        self._playback_task: Optional[asyncio.Task] = None
        self._utterance_queue: asyncio.Queue = asyncio.Queue()
        self._playback_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: Optional[asyncio.Queue] = None
        self._playback_interrupt = threading.Event()
        self._speaking = False
        self._policy = VOICE_INTERJECT_POLICY.lower()
        self._streaming_session: Optional[utils.PersonaPlexStreamingSession] = None
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
        self._playback_task = asyncio.create_task(self._playback_loop())
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
        tasks = [t for t in (self._listen_task, self._process_task, self._playback_task) if t is not None]
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

    async def say_chunks(self, chunks_iterator):
        """Asynchronously play an iterator of audio chunks."""
        for chunk in chunks_iterator:
            if self._stop_event.is_set():
                break
            self._playback_queue.put_nowait(chunk)
            # Match the playback rate roughly
            await asyncio.sleep(VOICE_CHUNK_SECONDS * 0.8)

    async def say_stream(self, sync_iterator):
        """Consume a synchronous audio generator and stream to the playback queue.
        
        This bridges the synchronous tts_stream generator to the async playback
        loop, allowing the agent to start speaking as soon as the first chunk
        is generated.
        """
        self._speaking = True
        await self._set_activity("speaking", "speaking")
        self._playback_interrupt.clear()
        
        loop = asyncio.get_running_loop()
        text_buffer = []
        
        def _run_gen():
            try:
                for item in sync_iterator:
                    if self._playback_interrupt.is_set() or self._stop_event.is_set():
                        break
                    
                    # Handle both single PCM chunks and (PCM, text) tuples
                    if isinstance(item, tuple):
                        chunk, piece = item
                    else:
                        chunk, piece = item, ""
                        
                    if piece:
                        text_buffer.append(piece)
                        # Log text pieces as they arrive for visibility
                        if len(text_buffer) % 5 == 0:
                            logger.info("Voice model (streaming): %s", "".join(text_buffer[-20:]))

                    if chunk is not None and chunk.size > 0:
                        loop.call_soon_threadsafe(self._playback_queue.put_nowait, chunk)
            except Exception as e:
                logger.error("say_stream generator error: %s", e)

        try:
            # Run the generator in a thread to keep the event loop free
            await asyncio.to_thread(_run_gen)
            
            final_text = "".join(text_buffer).strip()
            if final_text:
                await self.interaction_log_manager.append(f"Voice model (streaming): {final_text}")

            # Wait for playback queue to empty before finishing,
            # or until interrupted.
            while not self._playback_queue.empty() and not self._playback_interrupt.is_set():
                await asyncio.sleep(0.1)
                
            if self._playback_interrupt.is_set():
                await self._set_activity("interrupted", "playback interrupted")
            else:
                await self._set_activity("listening", "playback complete")
        finally:
            self._speaking = False
            # Invalidate the streaming session — tts_stream modified lm_gen state,
            # so the next session must start fresh with _restore_primed_state().
            self._streaming_session = None
            
            # Drain stale audio accumulated during TTS
            drained = 0
            if self._audio_queue is not None:
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                        drained += 1
                    except Exception:
                        break
            if drained:
                logger.debug("say_stream: drained %d stale audio chunks", drained)

    async def say_audio(self, pcm: "np.ndarray") -> None:
        """Play a pre-rendered continuous PCM array as one uninterrupted utterance.

        Preferred over say_chunks for TTS where all frames are already computed,
        because say_chunks plays frame-by-frame with gaps that break up speech.
        """
        if np is None or pcm.size == 0 or float(np.abs(pcm).max()) < 1e-5:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, pcm.astype(np.float32), VOICE_SAMPLE_RATE)
            self._speaking = True
            await self._set_activity("speaking", "speaking")
            await asyncio.to_thread(
                utils.play_wav_file_interruptible, tmp, self._playback_interrupt
            )
            self._speaking = False
            # Invalidate the streaming session — tts_stream modified lm_gen state,
            # so the next session must start fresh with _restore_primed_state().
            self._streaming_session = None
            # Drain stale audio that accumulated while the GPU was busy with TTS
            # so the model won't respond to audio that's already minutes old.
            drained = 0
            if self._audio_queue is not None:
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                        drained += 1
                    except Exception:
                        break
            if drained:
                logger.debug("say_audio: drained %d stale audio chunks after TTS playback", drained)
            await self._set_activity("listening", "playback complete")
        except Exception as e:
            self._speaking = False
            self._streaming_session = None
            logger.error("say_audio error: %s", e)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

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

    async def _playback_loop(self):
        """Continuously pulls chunks from playback queue and plays them."""
        import sounddevice as sd
        while not self._stop_event.is_set():
            try:
                # Use a small timeout to keep the loop responsive to stop events
                chunk = await asyncio.wait_for(self._playback_queue.get(), timeout=0.1)
                if chunk is None or chunk.size == 0:
                    continue
                
                # Check for silence muting (if all zeros, skip playback)
                if np.abs(chunk).max() < 1e-5:
                    continue

                # say_stream/say_audio already set _speaking=True, but we ensure it
                # here for chunks injected from other sources or if state drifted.
                was_speaking = self._speaking
                if not was_speaking:
                    self._speaking = True
                    await self._set_activity("speaking", "playback started")

                dev_idx = sd.default.device[1]
                logger.debug("Playback loop: playing chunk on device %s", dev_idx)
                
                # Calculate precise duration to avoid gaps or overlaps
                duration = len(chunk) / VOICE_SAMPLE_RATE
                
                sd.play(chunk, samplerate=VOICE_SAMPLE_RATE)
                
                # Sleep for slightly less than the duration to allow for event loop
                # overhead and ensure the next chunk is ready in time.
                # Factor reduced to 0.92 to be more proactive against gaps.
                await asyncio.sleep(duration * 0.92)
                
                # Only reset _speaking if the queue is actually empty, 
                # otherwise we are still in a continuous utterance.
                if self._playback_queue.empty():
                    self._speaking = False
                    await self._set_activity("listening", "playback finished")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Playback loop error: %s", e)
                self._speaking = False
                await asyncio.sleep(0.1)

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

                # VAD Muting: if it's not speech, feed silence to the model to avoid noise accumulation
                chunk_to_process = chunk if is_speech else np.zeros_like(chunk)

                # Full-duplex streaming path
                if self.personaplex_manager and not self._speaking:
                    if self._streaming_session is None:
                        self._streaming_session = self.personaplex_manager.create_session(
                            PERSONAPLEX_TEXT_PROMPT, PERSONAPLEX_VOICE_PROMPT
                        )
                        # Use the dedicated sequential executor to avoid jitter
                        await asyncio.get_running_loop().run_in_executor(
                            self.personaplex_manager.step_executor,
                            self._streaming_session.start
                        )
                    
                    # Capture local ref — say_audio() may set self._streaming_session = None
                    # concurrently, causing AttributeError if we use self._streaming_session.step
                    session = self._streaming_session
                    if session is None:
                        continue

                    # Use the dedicated sequential executor for every step
                    out_audio, out_text = await asyncio.get_running_loop().run_in_executor(
                        self.personaplex_manager.step_executor,
                        session.step,
                        chunk_to_process
                    )
                    
                    if out_text:
                        self._streaming_text_buffer += out_text
                        # Update rolling recent tokens for debug screen
                        self.state["recent_tokens"] = (self.state.get("recent_tokens", "") + out_text)[-100:]
                        # Periodically update the UI with what the model is transcribing/thinking
                        if len(self._streaming_text_buffer) % 20 == 0:
                            await self._set_activity("thinking", f"Model: {self._streaming_text_buffer[-40:]}")

                    if out_audio is not None and np.abs(out_audio).max() > 0.01:
                        # Buffer model audio; do NOT play immediately.
                        # Playing every frame as it arrives causes:
                        #   1. Filler audio during silence (model generates when fed zeros)
                        #   2. Double-playback (audio also replays at utterance boundary)
                        # Playback is handled once at utterance boundary in _process_loop.
                        self._streaming_audio_buffer.append(out_audio)

                if is_speech and self._speaking and self._policy == "hard":
                    # Only set+log once per actual interrupt (debounce: guard on is_set())
                    if not self._playback_interrupt.is_set():
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
                            combined_audio = np.concatenate(self._streaming_audio_buffer)
                            # Only queue if the model actually generated audible audio
                            if float(np.abs(combined_audio).max()) > 0.02:
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
                import traceback
                err_msg = str(exc) or type(exc).__name__
                tb = traceback.format_exc()
                logger.error("Voice listen loop error: %s\n%s", err_msg, tb)
                await self._set_activity("error", f"listen error: {err_msg[:80]}")
                # Reset broken session so start() is retried cleanly next chunk.
                self._streaming_session = None
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
                                utils.play_wav_file_interruptible,
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
                    elif control == "P":
                        await self._set_activity("passive", "Model chose passive mode (listening).")
                        await asyncio.sleep(0.5)
                        await self._set_activity("listening", "Resuming listen mode.")
                    else:
                        await self._set_activity("listening", "Model choosing next turn...")
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
                        result = await utils.run_personaplex_offline(
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
                            utils.play_wav_file_interruptible,
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
