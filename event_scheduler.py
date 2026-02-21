# event_scheduler.py

import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("autonomous_system.event_scheduler")


class EventScheduler:
    def __init__(self, state, interaction_log_manager, index_manager, runtime_manager=None):
        self.event_queue = asyncio.Queue()
        self.state = state
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager
        self.runtime_manager = runtime_manager
        self._active_tasks = set()
        self._stop_event = asyncio.Event()
        self.logger = logging.getLogger("autonomous_system.event_scheduler")

    async def start(self):
        """Starts the event scheduler."""
        self.logger.debug("Starting event scheduler")
        while not self._stop_event.is_set():
            event = await self.event_queue.get()
            if event is None:
                continue
            task = asyncio.create_task(self.handle_event(event))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            if self.runtime_manager is not None:
                self.runtime_manager.register_task(task)
            await asyncio.sleep(1)

    async def schedule_event(self, event):
        """Schedules an event."""
        self.logger.debug(f"Scheduling event: {event}")
        await self.event_queue.put(event)

    async def handle_event(self, event):
        """Handles an event."""
        event_type = event["type"]
        self.logger.info(f"Handling event: {event}")

        if event_type == "reminder":
            await self.interaction_log_manager.append(
                f"\nReminder: {event['message']}\n"
            )
        elif event_type == "lookup":
            keyword = event["keyword"]
            results = self.index_manager.search_context(keyword)
            self.logger.debug(f"Lookup results: {results}")
        elif event_type == "deferred_topic":
            topic = event["topic"]
            message = f"I've revisited {topic} and have more insight."
            await self.interaction_log_manager.append(f"\nThought: {message}\n")
        elif event_type == "rag_completed":
            message = "RAG completed. Scheduling training."
            training_event = {
                "type": "training",
                "message": message,
                "trigger_time": datetime.now() + timedelta(minutes=5),
            }
            await self.schedule_event(training_event)
        elif event_type == "training":
            await self.interaction_log_manager.append(
                f"\nTraining: {event['message']}\n"
            )

        self.state["next_event"] = (
            "Not scheduled" if self.event_queue.empty() else "Event pending"
        )

    async def shutdown(self):
        """Cancels in-flight scheduled event handlers."""
        self._stop_event.set()
        await self.event_queue.put(None)
        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
