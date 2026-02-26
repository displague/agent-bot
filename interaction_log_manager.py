# interaction_log_manager.py

import asyncio
import json
import logging
import os
from datetime import datetime
from config import INTERACTION_LOG_PATH

logger = logging.getLogger("autonomous_system.interaction_log_manager")


class InteractionLogManager:
    def __init__(self):
        self.interaction_log = []
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger("autonomous_system.interaction_log_manager")

    async def append(self, entry):
        """Appends an entry to the interaction log and persists it to file."""
        async with self.lock:
            self.interaction_log.append(entry)
            self.logger.debug(f"Appended entry: {entry}")

            # Persist to hard log file
            try:
                os.makedirs(os.path.dirname(INTERACTION_LOG_PATH), exist_ok=True)
                with open(INTERACTION_LOG_PATH, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "event": entry,
                    }
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to persist log entry: {e}")

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

    async def get_entries_since(self, offset):
        """Returns new log entries and the latest offset."""
        async with self.lock:
            safe_offset = max(0, min(offset, len(self.interaction_log)))
            entries = self.interaction_log[safe_offset:]
            return entries, len(self.interaction_log)
