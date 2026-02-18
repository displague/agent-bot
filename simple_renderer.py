import asyncio
import logging

logger = logging.getLogger("autonomous_system.simple_renderer")


class SimpleRenderer:
    """Cross-platform fallback renderer when curses is unavailable."""

    def __init__(self, state, interaction_queue, interaction_log_manager):
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self._stop = False

    async def start(self):
        print("Simple mode active (no curses). Type /quit to exit.")
        while not self._stop:
            try:
                line = await asyncio.to_thread(input, "Input> ")
                if not line:
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
            except (EOFError, KeyboardInterrupt):
                self._stop = True
            except Exception as e:
                logger.error("Simple renderer error: %s", e)
                await asyncio.sleep(0.1)
