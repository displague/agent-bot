import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Set


class RuntimeManager:
    """Owns thread executors and tracks async tasks for coordinated shutdown."""

    def __init__(self, llm_workers: int = 1, io_workers: int = 2):
        self.llm_executor = ThreadPoolExecutor(max_workers=llm_workers)
        self.io_executor = ThreadPoolExecutor(max_workers=io_workers)
        self._tasks: Set[asyncio.Task] = set()

    def register_task(self, task: asyncio.Task) -> asyncio.Task:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all_tasks(self, timeout_seconds: float = 5.0) -> None:
        pending = [t for t in self._tasks if not t.done()]
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)

    def shutdown_executors(self) -> None:
        self.llm_executor.shutdown(wait=False, cancel_futures=True)
        self.io_executor.shutdown(wait=False, cancel_futures=True)
