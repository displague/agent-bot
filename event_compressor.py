# event_compressor.py

import asyncio
import json
import os
from datetime import datetime
import logging

from config import (
    COMPRESSED_LOG_PATH,
    EVENT_COMPRESS_MAX_ENTRIES,
    INTERACTION_LOG_PATH,
)

logger = logging.getLogger("autonomous_system.event_compressor")


class EventCompressor:
    """
    Compresses events periodically, summarizing interactions for future training.
    """

    def __init__(self, llama_manager, event_scheduler, io_executor=None, state=None):
        self.llama_manager = llama_manager
        self.event_scheduler = event_scheduler
        self.io_executor = io_executor
        self.state = state or {}
        self.logger = logging.getLogger("autonomous_system.event_compressor")

    async def start(self):
        """Starts the periodic event compression."""
        self.logger.debug("Starting periodic event compression...")
        while True:
            await self.compress_events()
            await asyncio.sleep(3600)  # Run every hour

    async def compress_events(self):
        """Compresses events."""
        self.logger.debug("Starting event compression")
        if self.state.get("is_processing", False):
            self.logger.debug("Skipping compression while user interaction is processing.")
            return
        if not os.path.exists(INTERACTION_LOG_PATH):
            self.logger.debug("No logs to compress.")
            return
        try:
            logs = []
            skipped_lines = 0
            with open(INTERACTION_LOG_PATH, "r", encoding="utf-8") as log_file:
                for line in log_file:
                    if not line.strip():
                        continue
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        skipped_lines += 1
            if skipped_lines:
                self.logger.warning(
                    "Skipped %s malformed interaction log lines during compression",
                    skipped_lines,
                )
            if len(logs) > EVENT_COMPRESS_MAX_ENTRIES:
                logs = logs[-EVENT_COMPRESS_MAX_ENTRIES:]
            if not logs:
                self.logger.debug("No events to compress.")
                return

            events_text = ""
            for entry in logs:
                event_str = entry.get("event", "")
                if not event_str:
                    # Fallback for old schema
                    input_text = entry.get("input", "")
                    output_text = entry.get("output", "")
                    if input_text or output_text:
                        events_text += f"Task: {input_text}\nThought: {output_text}\n"
                    continue

                if event_str.startswith("Input: "):
                    events_text += f"Task: {event_str[len('Input: '):]}\n"
                elif event_str.startswith("Thought: "):
                    events_text += f"Thought: {event_str[len('Thought: '):]}\n"
                elif event_str.startswith("Voice model"):
                    events_text += f"Voice: {event_str}\n"

            if not events_text.strip():
                self.logger.debug("No events to compress.")
                return

            # Use Llama model to generate a summary
            prompt = f"""Summarize the following interactions for future reference:

{events_text}

Summary:"""
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                self.io_executor, self.llama_manager.llm_call, prompt
            )
            self.logger.debug(f"Summary generated: {summary}")

            # Save the compressed summary
            compressed_entry = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
            }
            with open(COMPRESSED_LOG_PATH, "a") as comp_log_file:
                comp_log_file.write(json.dumps(compressed_entry) + "\n")
            self.logger.info("Event compression completed")

            # Schedule a RAG event
            rag_event = {"type": "rag_completed", "trigger_time": datetime.now()}
            await self.event_scheduler.schedule_event(rag_event)

        except Exception as e:
            self.logger.error(f"Error in compress_events: {e}")
            await asyncio.sleep(1)
