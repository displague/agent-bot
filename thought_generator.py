# thought_generator.py

import asyncio
import random
from datetime import datetime
import logging

from config import DAILY_SLEEP_START, DAILY_SLEEP_END
from functional_agent import FunctionalAgent

logger = logging.getLogger("autonomous_system.thought_generator")


class ThoughtGenerator:
    """
    Generates autonomous thoughts periodically, using the FunctionalAgent for multi-phase processing.
    """

    def __init__(self, state, llama_manager, interaction_log_manager, event_scheduler):
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.event_scheduler = event_scheduler
        self.functional_agent = FunctionalAgent(self.llama_manager)
        self.logger = logging.getLogger("autonomous_system.thought_generator")

    async def start(self):
        """Starts the autonomous thought generation."""
        self.logger.debug("Starting autonomous thought generation...")
        while True:
            current_hour = datetime.now().hour
            if DAILY_SLEEP_START <= current_hour or current_hour < DAILY_SLEEP_END:
                self.state["is_sleeping"] = True
                await asyncio.sleep(random.uniform(5, 10))
            else:
                self.state["is_sleeping"] = False
                self.logger.debug("Generating new autonomous thoughts")
                tasks = [asyncio.create_task(self.generate_thought()) for _ in range(3)]
                await asyncio.gather(*tasks)
                await asyncio.sleep(random.uniform(1, 3))

    async def generate_thought(self):
        """Generates a single thought."""
        self.state["ongoing_thoughts"] += 1
        thought_prompt = f"Autonomous thought at {datetime.now().strftime('%H:%M:%S')}"
        try:
            response = await self.functional_agent.handle_request(thought_prompt)
            self.state.setdefault("current_thoughts", []).append(response)
            await self.interaction_log_manager.append(f"Thought: {response}")
            await asyncio.sleep(random.uniform(1, 3))
        except Exception as e:
            self.logger.error(f"Error in generate_thought: {e}")
            await asyncio.sleep(1)
        finally:
            self.state["ongoing_thoughts"] = max(0, self.state["ongoing_thoughts"] - 1)
            if (
                "current_thoughts" in self.state
                and response in self.state["current_thoughts"]
            ):
                self.state["current_thoughts"].remove(response)
