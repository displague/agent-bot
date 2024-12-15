# event_compressor.py

import asyncio
import json
import os
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

from config import COMPRESSED_LOG_PATH, HARD_LOG_PATH, MAX_WORKERS

logger = logging.getLogger("autonomous_system.event_compressor")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class EventCompressor:
    """
    Compresses events periodically, summarizing interactions for future training.
    """

    def __init__(self, llama_manager, event_scheduler):
        self.llama_manager = llama_manager
        self.event_scheduler = event_scheduler
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
        if not os.path.exists(HARD_LOG_PATH):
            self.logger.debug("No logs to compress.")
            return
        try:
            with open(HARD_LOG_PATH, "r") as log_file:
                logs = [json.loads(line) for line in log_file if line.strip()]
            if not logs:
                self.logger.debug("No events to compress.")
                return

            events_text = ""
            for entry in logs:
                input_text = entry.get("input", "")
                output_text = entry.get("output", "")
                events_text += f"Task: {input_text}\nThought: {output_text}\n"

            if not events_text.strip():
                self.logger.debug("No events to compress.")
                return

            # Use Llama model to generate a summary
            prompt = f"""Summarize the following interactions for future reference:

{events_text}

Summary:"""
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                executor, self.llama_manager.llm_call, prompt
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
