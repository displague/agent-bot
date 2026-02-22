# interaction_processor.py

import asyncio
import json
import random
from datetime import datetime
import logging

from functional_agent import FunctionalAgent
from config import INTERACTION_LOG_PATH, INTERACTION_PROCESS_TIMEOUT_SECONDS
from utils import extract_text_features, extract_audio_features

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
    ):
        self.interaction_queue = interaction_queue
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager
        self.functional_agent = FunctionalAgent(self.llama_manager, state=self.state)
        self.logger = logging.getLogger("autonomous_system.interaction_processor")
        self._stop_event = asyncio.Event()

    async def start(self):
        """Starts the interaction processing."""
        self.logger.debug("Starting interaction processing...")
        while not self._stop_event.is_set():
            if not self.interaction_queue.empty():
                try:
                    interaction = self.interaction_queue.get_nowait()
                    user_input = interaction.get("input", "")
                    audio_waveform = interaction.get("audio_waveform", None)
                    self.state["is_processing"] = True
                    self.state["last_processing_input"] = (user_input or "")[:240]
                    self.state["last_processing_started_at"] = datetime.now().isoformat()
                    self.state["last_processing_status"] = "running"
                    self.state["last_processing_error"] = ""

                    self.logger.info(f"Processing interaction: {user_input}")

                    if audio_waveform is not None:
                        audio_features = extract_audio_features(audio_waveform)

                    # Process request with multi-phase approach
                    response = await asyncio.wait_for(
                        self.functional_agent.handle_request(user_input),
                        timeout=INTERACTION_PROCESS_TIMEOUT_SECONDS,
                    )

                    self.logger.info(f"Response: {response}")

                    await self.interaction_log_manager.append(f"Thought: {response}")
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "input": user_input,
                        "output": response,
                    }
                    with open(INTERACTION_LOG_PATH, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")

                    self.index_manager.index_interaction(log_entry)
                    self.state["last_processing_status"] = "ok"
                    self.state["last_processing_error"] = ""
                    self.state["unprocessed_interactions"] = max(
                        0, self.state["unprocessed_interactions"] - 1
                    )
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
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
