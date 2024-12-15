# interaction_processor.py

import asyncio
import json
import random
from datetime import datetime
import logging

from functional_agent import FunctionalAgent
from config import HARD_LOG_PATH
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
        self.functional_agent = FunctionalAgent(self.llama_manager)
        self.logger = logging.getLogger("autonomous_system.interaction_processor")

    async def start(self):
        """Starts the interaction processing."""
        self.logger.debug("Starting interaction processing...")
        while True:
            if not self.interaction_queue.empty():
                interaction = self.interaction_queue.get()
                try:
                    user_input = interaction.get("input", "")
                    audio_waveform = interaction.get("audio_waveform", None)

                    self.logger.info(f"Processing interaction: {user_input}")

                    if audio_waveform is not None:
                        audio_features = extract_audio_features(audio_waveform)

                    # Process request with multi-phase approach
                    response = await self.functional_agent.handle_request(user_input)

                    self.logger.info(f"Response: {response}")

                    await self.interaction_log_manager.append(f"Thought: {response}")
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "input": user_input,
                        "output": response,
                    }
                    with open(HARD_LOG_PATH, "a") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")

                    self.index_manager.index_interaction(log_entry)
                    self.state["unprocessed_interactions"] = max(
                        0, self.state["unprocessed_interactions"] - 1
                    )
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                except Exception as e:
                    self.logger.error(f"Error processing interaction: {e}")
            else:
                await asyncio.sleep(0.1)
