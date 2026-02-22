# interaction_processor.py

import asyncio
import json
import random
import os
import tempfile
import logging
from datetime import datetime

import numpy as np
import soundfile as sf

from functional_agent import FunctionalAgent
from config import (
    INTERACTION_LOG_PATH, 
    INTERACTION_PROCESS_TIMEOUT_SECONDS,
    PERSONAPLEX_VERBAL_FILLERS,
    PERSONAPLEX_VOICE_PROMPT,
    VOICE_SAMPLE_RATE,
    VOICE_OFFLINE_INFER_TIMEOUT_SECONDS,
)
from utils import extract_text_features, extract_audio_features, run_personaplex_offline, play_wav_file_interruptible, transcribe_audio

logger = logging.getLogger("autonomous_system.interaction_processor")


class InteractionProcessor:
    """
    Processes interactions using the FunctionalAgent, handling each interaction in phases.
    """

    def __init__(
        self,
        interaction_queue,
        state,
        llama_manager,
        interaction_log_manager,
        index_manager,
        voice_loop=None,
        personaplex_manager=None,
    ):
        self.interaction_queue = interaction_queue
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager
        self.voice_loop = voice_loop
        self.personaplex_manager = personaplex_manager
        self._processing_lock = asyncio.Lock()
        self.functional_agent = FunctionalAgent(self.llama_manager, state=self.state)
        self.logger = logging.getLogger("autonomous_system.interaction_processor")
        self._stop_event = asyncio.Event()

    async def _trigger_verbal_filler(self):
        """Immediately triggers a reflexive verbal filler through PersonaPlex."""
        if not self.voice_loop or not PERSONAPLEX_VERBAL_FILLERS:
            return
        
        filler = random.choice(PERSONAPLEX_VERBAL_FILLERS)
        self.logger.info(f"Triggering verbal filler: {filler}")
        
        try:
            with tempfile.TemporaryDirectory(prefix="filler_") as tmpdir:
                input_wav = os.path.join(tmpdir, "silence.wav")
                output_wav = os.path.join(tmpdir, "filler.wav")
                
                # Create a tiny silence file for offline mode to generate from prompt
                duration = 0.5
                samples = int(duration * VOICE_SAMPLE_RATE)
                silence = np.zeros(samples, dtype=np.float32)
                sf.write(input_wav, silence, VOICE_SAMPLE_RATE)
                
                if self.personaplex_manager:
                    await self.personaplex_manager.infer_async(
                        text_prompt=filler,
                        voice_prompt_path=PERSONAPLEX_VOICE_PROMPT,
                        input_wav_path=input_wav,
                        output_wav_path=output_wav
                    )
                else:
                    await run_personaplex_offline(
                        input_wav,
                        output_wav,
                        text_prompt=filler,
                        voice_prompt=PERSONAPLEX_VOICE_PROMPT,
                        timeout_seconds=VOICE_OFFLINE_INFER_TIMEOUT_SECONDS,
                    )
                
                if os.path.exists(output_wav):
                    await asyncio.to_thread(
                        play_wav_file_interruptible,
                        output_wav,
                        getattr(self.voice_loop, "_playback_interrupt", None)
                    )
        except Exception as e:
            self.logger.error(f"Failed to trigger verbal filler: {e}")

    async def start(self):
        """Starts the interaction processing."""
        self.logger.debug("Starting interaction processing...")
        while not self._stop_event.is_set():
            if not self.interaction_queue.empty():
                async with self._processing_lock:
                    try:
                        interaction = self.interaction_queue.get_nowait()
                        user_input = interaction.get("input", "")
                        audio_waveform = interaction.get("audio_waveform", None)
                        
                        # If audio is present but input is empty, transcribe first
                        if audio_waveform is not None and not user_input:
                            self.state["processing_phase"] = "Transcribing"
                            self.logger.info("Transcribing audio waveform...")
                            user_input = await transcribe_audio(audio_waveform)
                            self.logger.info(f"Transcribed: {user_input}")
                            if not user_input:
                                self.logger.info("Empty transcription, skipping interaction.")
                                self.state["unprocessed_interactions"] = max(0, self.state["unprocessed_interactions"] - 1)
                                continue
                            await self.interaction_log_manager.append(f"Voice Input: {user_input}")

                        self.state["is_processing"] = True
                        self.state["last_processing_input"] = (user_input or "")[:240]
                        self.state["last_processing_started_at"] = datetime.now().isoformat()
                        self.state["last_processing_status"] = "running"
                        self.state["last_processing_error"] = ""

                        self.logger.info(f"Processing interaction: {user_input}")

                        if audio_waveform is not None:
                            audio_features = extract_audio_features(audio_waveform)

                        # Trigger reflexive verbal filler concurrently with deep reasoning
                        filler_task = asyncio.create_task(self._trigger_verbal_filler())

                        # Process request with multi-phase approach
                        response = await asyncio.wait_for(
                            self.functional_agent.handle_request(user_input),
                            timeout=INTERACTION_PROCESS_TIMEOUT_SECONDS,
                        )

                        await filler_task
                        self.logger.info(f"Response: {response}")

                        # Speak final deep response aloud
                        try:
                            with tempfile.TemporaryDirectory(prefix="final_") as tmpdir:
                                input_wav = os.path.join(tmpdir, "silence.wav")
                                output_wav = os.path.join(tmpdir, "final.wav")
                                duration = 0.5
                                samples = int(duration * VOICE_SAMPLE_RATE)
                                silence = np.zeros(samples, dtype=np.float32)
                                sf.write(input_wav, silence, VOICE_SAMPLE_RATE)
                                
                                if self.personaplex_manager:
                                    await self.personaplex_manager.infer_async(
                                        text_prompt=response,
                                        voice_prompt_path=PERSONAPLEX_VOICE_PROMPT,
                                        input_wav_path=input_wav,
                                        output_wav_path=output_wav
                                    )
                                else:
                                    await run_personaplex_offline(
                                        input_wav,
                                        output_wav,
                                        text_prompt=response,
                                        voice_prompt=PERSONAPLEX_VOICE_PROMPT,
                                        timeout_seconds=VOICE_OFFLINE_INFER_TIMEOUT_SECONDS,
                                    )
                                
                                if os.path.exists(output_wav):
                                    await asyncio.to_thread(
                                        play_wav_file_interruptible,
                                        output_wav,
                                        getattr(self.voice_loop, "_playback_interrupt", None)
                                    )
                        except Exception as e:
                            self.logger.error(f"Failed to speak final response: {e}")

                        await self.interaction_log_manager.append(f"Thought: {response}")
                        self.index_manager.index_interaction({"input": user_input, "output": response})
                        self.state["last_processing_status"] = "ok"
                        self.state["last_processing_error"] = ""
                        self.state["unprocessed_interactions"] = max(
                            0, self.state["unprocessed_interactions"] - 1
                        )
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    except asyncio.QueueEmpty:
                        pass
                    except asyncio.TimeoutError:
                        msg = (
                            f"interaction timeout after {INTERACTION_PROCESS_TIMEOUT_SECONDS}s"
                        )
                        self.logger.error(msg)
                        await self.interaction_log_manager.append(f"System: {msg}")
                        self.state["last_processing_status"] = "timeout"
                        self.state["last_processing_error"] = msg
                        self.state["unprocessed_interactions"] = max(
                            0, self.state["unprocessed_interactions"] - 1
                        )
                    except Exception as e:
                        self.logger.error(f"Error processing interaction: {e}")
                        await self.interaction_log_manager.append(
                            f"System: interaction failed ({str(e)[:120]})"
                        )
                        self.state["last_processing_status"] = "error"
                        self.state["last_processing_error"] = str(e)[:240]
                        self.state["unprocessed_interactions"] = max(
                            0, self.state["unprocessed_interactions"] - 1
                        )
                    finally:
                        self.state["is_processing"] = False
            else:
                await asyncio.sleep(0.1)

    def request_stop(self):
        self._stop_event.set()
