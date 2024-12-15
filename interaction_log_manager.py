# interaction_log_manager.py

import asyncio
import logging

logger = logging.getLogger("autonomous_system.interaction_log_manager")


class InteractionLogManager:
    def __init__(self):
        self.interaction_log = []
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("autonomous_system.interaction_log_manager")

    async def append(self, entry):
        """Appends an entry to the interaction log."""
        async with self.lock:
            self.interaction_log.append(entry)
            self.logger.debug(f"Appended entry: {entry}")

    async def get_display_log(self, max_items, scroll_offset=0):
        """Gets a portion of the log for display."""
        async with self.lock:
            start_index = max(
                0, len(self.interaction_log) - (max_items + scroll_offset)
            )
            end_index = len(self.interaction_log) - scroll_offset
            display_log = self.interaction_log[start_index:end_index]
            self.logger.debug(f"Getting log from {start_index} to {end_index}")
            return display_log
