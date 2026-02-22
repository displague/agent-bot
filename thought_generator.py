# thought_generator.py

import asyncio
import random
from datetime import datetime
import logging

from config import (
    DAILY_SLEEP_END,
    DAILY_SLEEP_START,
    THOUGHT_MAX_CONCURRENCY,
    THOUGHT_MIN_INTERVAL_SECONDS,
)
from functional_agent import FunctionalAgent

logger = logging.getLogger("autonomous_system.thought_generator")


class ThoughtGenerator:
    """
    Generates autonomous thoughts periodically, using the FunctionalAgent for multi-phase processing.
    """

    def __init__(self, state, llama_manager, interaction_log_manager, event_scheduler, interaction_processor=None):
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.event_scheduler = event_scheduler
        self.interaction_processor = interaction_processor
        self.functional_agent = FunctionalAgent(self.llama_manager, state=self.state)
        self.logger = logging.getLogger("autonomous_system.thought_generator")
        self._semaphore = asyncio.Semaphore(THOUGHT_MAX_CONCURRENCY)
        self._stop_event = asyncio.Event()

    async def start(self):
        """Starts the autonomous thought generation."""
        self.logger.debug("Starting autonomous thought generation...")
        while not self._stop_event.is_set():
            current_hour = datetime.now().hour
            is_sleep_time = (DAILY_SLEEP_START <= current_hour or current_hour < DAILY_SLEEP_END)
            
            if is_sleep_time and not self.state.get("manual_wake"):
                self.state["is_sleeping"] = True
                await asyncio.sleep(random.uniform(5, 10))
            else:
                self.state["is_sleeping"] = False
                self.logger.debug("Generating new autonomous thoughts")
                await self.generate_thought()
                await asyncio.sleep(max(THOUGHT_MIN_INTERVAL_SECONDS, random.uniform(1, 3)))

    async def generate_thought(self):
        """Generates a single thought."""
        async with self._semaphore:
            # Also synchronize with the main interaction processor to avoid GPU contention
            lock_ctx = (
                self.interaction_processor._processing_lock 
                if self.interaction_processor else asyncio.Lock()
            )
            
            async with lock_ctx:
                self.state["ongoing_thoughts"] += 1
                thought_prompt = f"Autonomous thought at {datetime.now().strftime('%H:%M:%S')}"
                response = None
                try:
                    response = await self.functional_agent.handle_request(thought_prompt)
                    self.state.setdefault("current_thoughts", []).append(response)
                    while len(self.state["current_thoughts"]) > 5:
                        self.state["current_thoughts"].pop(0)
                    await self.interaction_log_manager.append(f"Thought: {response}")
                    await asyncio.sleep(random.uniform(1, 3))
                except Exception as e:
                    self.logger.error(f"Error in generate_thought: {e}")
                    await asyncio.sleep(1)
                finally:
                    self.state["ongoing_thoughts"] = max(0, self.state["ongoing_thoughts"] - 1)

    def request_stop(self):
        self._stop_event.set()
