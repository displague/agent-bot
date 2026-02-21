import asyncio
import logging

from config import VOICE_CAPTURE_KEY_LABEL
from smoke_test_runner import (
    run_deterministic_smoke,
    run_model_smoke,
    summarize_smoke_result,
)

logger = logging.getLogger("autonomous_system.simple_renderer")


class SimpleRenderer:
    """Cross-platform fallback renderer when curses is unavailable."""

    def __init__(
        self,
        state,
        interaction_queue,
        interaction_log_manager,
        functional_agent=None,
    ):
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self.functional_agent = functional_agent
        self._stop = False
        self._last_log_offset = 0
        self._log_pump_task = None

    async def start(self):
        print("Simple mode active (no curses).")
        print("Type /help for commands. Type /quit to exit.")
        self._log_pump_task = asyncio.create_task(self._pump_logs())
        try:
            while not self._stop:
                try:
                    line = await asyncio.to_thread(input, "Input> ")
                    if not line:
                        continue
                    if line.startswith("/"):
                        await self.handle_command(line)
                        continue
                    if line.strip().lower() in {"/quit", "/exit"}:
                        self._stop = True
                        break
                    self.interaction_queue.put_nowait(
                        {
                            "input": line,
                            "private_notes": "",
                            "audio_waveform": None,
                        }
                    )
                    self.state["unprocessed_interactions"] += 1
                    await self.interaction_log_manager.append(f"Input: {line}")
                    print("Queued input. Waiting for response...")
                except (EOFError, KeyboardInterrupt):
                    self._stop = True
                except Exception as e:
                    logger.error("Simple renderer error: %s", e)
                    await asyncio.sleep(0.1)
        finally:
            if self._log_pump_task is not None:
                self._log_pump_task.cancel()
                await asyncio.gather(self._log_pump_task, return_exceptions=True)

    async def _pump_logs(self):
        while not self._stop:
            try:
                entries, next_offset = await self.interaction_log_manager.get_entries_since(
                    self._last_log_offset
                )
                self._last_log_offset = next_offset
                for entry in entries:
                    print(f"[log] {entry}")
            except Exception as e:
                logger.error("Simple renderer log pump error: %s", e)
            await asyncio.sleep(0.2)

    async def handle_command(self, command: str):
        cmd = command.strip().lower()
        if cmd in {"/quit", "/exit"}:
            self._stop = True
            return
        if cmd == "/help":
            print("Commands: /help, /smoke, /smoke-model, /smoke-all, /quit")
            print(f"Voice capture is available in curses mode via {VOICE_CAPTURE_KEY_LABEL}.")
            return
        if cmd in {"/smoke", "/smoke-all"}:
            deterministic_result = await run_deterministic_smoke()
            deterministic_summary = summarize_smoke_result(deterministic_result)
            print(deterministic_summary)
            await self.interaction_log_manager.append(deterministic_summary)
        if cmd in {"/smoke-model", "/smoke-all"}:
            if self.functional_agent is None:
                msg = "SMOKE model: FAIL functional agent unavailable"
                print(msg)
                await self.interaction_log_manager.append(msg)
                return
            print("Running model smoke test...")
            model_result = await run_model_smoke(self.functional_agent)
            model_summary = summarize_smoke_result(model_result)
            print(model_summary)
            await self.interaction_log_manager.append(model_summary)
            return
        if cmd not in {"/smoke", "/smoke-all"}:
            print(f"Unknown command: {command}. Try /help")
