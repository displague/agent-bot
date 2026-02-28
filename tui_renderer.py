# tui_renderer.py

import asyncio
import curses
import textwrap
from queue import Queue, Empty as QueueEmpty
import logging
import os
import subprocess
import time
from datetime import datetime

from config import (
    PERSONAPLEX_VOICE_PROMPT,
    LLM_DIAG_TIMEOUT_SECONDS,
    VOICE_AUTO_START_ON_LAUNCH,
    VOICE_CAPTURE_KEY_CODE,
    VOICE_CAPTURE_KEY_LABEL,
)
from smoke_test_runner import (
    run_deterministic_smoke,
    run_model_smoke,
    summarize_smoke_result,
)
from process_utils import force_exit_now
from utils import _resolve_personaplex_python
from voice_loop import VoiceLoop

logger = logging.getLogger("autonomous_system.tui_renderer")


class TUIRenderer:
    COMMANDS = [
        "/help",
        "/model",
        "/llm-status",
        "/llm-diagnose",
        "/smoke",
        "/smoke-model",
        "/smoke-all",
        "/voice-start",
        "/voice-stop",
        "/voice-status",
        "/voice-diagnose",
        "/voice-test-tone",
        "/voice-devices",
        "/voice-set-device",
        "/voice-say",
        "/voice-hear",
        "/voice-hear-file",
        "/set-persona",
        "/voice-optimize",
        "/logic-reload",
        "/wake",
        "/sleep",
        "/force-quit",
        "/quit",
        "/exit",
    ]

    def __init__(
        self,
        stdscr,
        state,
        interaction_queue,
        interaction_log_manager,
        interaction_processor=None,
        voice_loop=None,
    ):
        self.logger = logging.getLogger("autonomous_system.tui_renderer")
        self.stdscr = stdscr
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self.interaction_processor = interaction_processor
        self.debug_queue = Queue()
        self.active_screen = 1
        self.input_buffer = ""
        self.debug_log = []
        self.scroll_offset = 0
        self.audio_waveform = None
        self._stop = False
        self._refresh_interval_seconds = 0.05
        self._voice_loop = voice_loop or VoiceLoop(state, interaction_log_manager)
        self.input_history = []
        self.history_index = -1
        self.cursor_pos = 0
        self.state.setdefault("is_listening", False)
        self.state.setdefault("voice_mode", "offline-disabled")
        self.state.setdefault("voice_server_state", "stopped")
        self.state.setdefault("voice_session_state", "disconnected")
        self.state.setdefault("voice_activity_state", "idle")
        self.state.setdefault("voice_last_event", "none")

    async def start(self):
        """Starts the TUI rendering."""
        self.logger.debug("Starting TUI rendering...")
        curses.curs_set(1)
        self.stdscr.nodelay(True)
        await self.interaction_log_manager.append(
            f"Type /help for commands. {VOICE_CAPTURE_KEY_LABEL} starts offline voice mode. Esc toggles debug. Ctrl+D exits."
        )
        if VOICE_AUTO_START_ON_LAUNCH:
            await self._set_voice_state(
                server="starting",
                session="disconnected",
                activity="starting",
                event="Auto-starting voice loop at launch",
            )
            await self.handle_command("/voice-start")
        try:
            while not self._stop:
                try:
                    await self.render()
                except Exception as e:
                    self.logger.exception("TUI render loop error: %s", e)
                    self.show_footer_message(f"TUI error: {str(e)[:120]}")
                    await self.interaction_log_manager.append(f"TUI error: {e}")
                await asyncio.sleep(self._refresh_interval_seconds)
        finally:
            await self._voice_loop.stop()

    async def _set_voice_state(
        self, *, server=None, session=None, activity=None, event=None
    ):
        if server is not None:
            self.state["voice_server_state"] = server
        if session is not None:
            self.state["voice_session_state"] = session
        if activity is not None:
            self.state["voice_activity_state"] = activity
        if event:
            self.state["voice_last_event"] = event
            await self.interaction_log_manager.append(f"Voice: {event}")
            self.show_footer_message(f"Voice: {event}")

    def _voice_indicator(self) -> str:
        activity = self.state.get("voice_activity_state", "idle")
        mapping = {
            "listening": "L",
            "thinking": "T",
            "speaking": "S",
            "passive": "P",
            "interrupted": "!",
            "idle": "-",
            "starting": "~",
            "error": "E",
            "failed": "E",
        }
        base = mapping.get(activity, "?")
        blink = activity in {"thinking", "interrupted"}
        if blink and (time.monotonic() * 2) % 2 < 1:
            return " "
        return base

    def _safe_addstr(self, y, x, text, attr=0):
        max_y, max_x = self.stdscr.getmaxyx()
        if y < 0 or x < 0 or y >= max_y or max_x <= 1:
            return
        clipped = (text or "")[: max_x - x - 1]
        try:
            if attr:
                self.stdscr.addstr(y, x, clipped, attr)
            else:
                self.stdscr.addstr(y, x, clipped)
        except curses.error:
            return

    def render_status_bar(self):
        """Renders the status bar."""
        max_y, max_x = self.stdscr.getmaxyx()
        sleep_status = "SLEEPING" if self.state["is_sleeping"] else "ACTIVE"
        listening_status = " LISTENING" if self.state.get("is_listening", False) else ""
        processing_status = (
            " PROCESSING" if self.state.get("is_processing", False) else ""
        )
        voice_mode = self.state.get("voice_mode", "unknown")
        voice_server_state = self.state.get("voice_server_state", "unknown")
        voice_session_state = self.state.get("voice_session_state", "unknown")
        voice_activity_state = self.state.get("voice_activity_state", "unknown")
        voice_last_event = self.state.get("voice_last_event", "none")
        voice_indicator = self._voice_indicator()
        status_bar = (
            f" Status: {sleep_status}{listening_status}{processing_status} | Unprocessed: {self.state['unprocessed_interactions']} "
            f"| Thoughts: {self.state['ongoing_thoughts']} | Next event: {self.state['next_event']} "
            f"| V:{voice_indicator} "
            f"| Voice: {voice_mode}/{voice_server_state}/{voice_session_state}/{voice_activity_state} "
            f"| VLast: {voice_last_event} "
        )
        self._safe_addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)
        try:
            self.stdscr.clrtoeol()
        except curses.error:
            pass

    async def render(self):
        """Renders the main screen or debug screen."""
        try:
            self.stdscr.erase()
        except curses.error:
            pass
        self.render_status_bar()

        if self.active_screen == 1:
            await self.render_main_screen()
        elif self.active_screen == 2:
            self.render_debug_screen()

        self.render_input_line()
        try:
            self.stdscr.refresh()
        except curses.error:
            pass

        await self.handle_input()
        self.process_debug_queue()

    async def render_main_screen(self):
        """Renders the main screen with correct scrolling logic."""
        max_y, max_x = self.stdscr.getmaxyx()
        current_y = 1

        # Room for thought lines
        current_thoughts = self.state.get("current_thoughts", [])
        for thought in current_thoughts:
            wrapped_thought = textwrap.wrap(f"Thought: {thought}", max_x)
            for line in wrapped_thought:
                if current_y >= max_y - 3:
                    break
                self._safe_addstr(current_y, 0, line)
                current_y += 1

        # Remaining room for interaction log
        log_height = max_y - current_y - 3
        if log_height <= 0:
            return

        # Fetch enough entries to potentially fill the screen, considering wraps
        # We fetch more than log_height just in case
        display_log = await self.interaction_log_manager.get_display_log(
            max_items=log_height + 50, scroll_offset=self.scroll_offset
        )

        # We render from bottom to top to fill the screen correctly
        rendered_lines = []
        for interaction in reversed(display_log):
            wrapped = textwrap.wrap(interaction, max_x)
            for line in reversed(wrapped):
                rendered_lines.insert(0, line)
                if len(rendered_lines) >= log_height:
                    break
            if len(rendered_lines) >= log_height:
                break

        for line in rendered_lines:
            self._safe_addstr(current_y, 0, line)
            current_y += 1

    def render_debug_screen(self):
        """Renders the debug screen with enhanced telemetry."""
        max_y, max_x = self.stdscr.getmaxyx()
        current_y = 1

        # Telemetry Summary
        vram = self.state.get("vram_gb", 0.0)
        inf_ms = self.state.get("inference_ms", 0.0)
        recent_toks = self.state.get("recent_tokens", "")

        debug_lines = [
            f"Debug | Loading: {self.state.get('loading_stage', 'Unknown')}",
            f"Debug | Phase: {self.state.get('processing_phase', 'idle')} | VRAM: {vram:.2f} GB | Inf: {inf_ms:.1f} ms/frame",
            f"Debug | Voice: {self.state.get('voice_mode')}/{self.state.get('voice_server_state')}/{self.state.get('voice_activity_state')}",
            f"Debug | Tokens: {recent_toks[-max_x+10:]}",  # Show last visible segment
            f"Debug | Last Status: {self.state.get('last_processing_status', '')} | Error: {self.state.get('last_processing_error', '')}",
        ]
        for item in debug_lines:
            self._safe_addstr(current_y, 0, item[:max_x])
            current_y += 1
            if current_y >= max_y - 3:
                break

        # Room for debug log entries
        max_log_lines = max_y - current_y - 3
        if max_log_lines <= 0:
            return

        display_log = self.debug_log[
            -(max_log_lines + self.scroll_offset) : (
                -self.scroll_offset if self.scroll_offset > 0 else None
            )
        ]
        for debug_message in display_log:
            wrapped_debug = textwrap.wrap(debug_message, max_x)
            for line in wrapped_debug:
                if current_y >= max_y - 3:
                    break
                self._safe_addstr(current_y, 0, line)
                current_y += 1

    def render_input_line(self):
        """Renders the input line with cursor support."""
        max_y, max_x = self.stdscr.getmaxyx()
        # Header "Input: " takes 7 chars
        max_chars = max(0, max_x - 8)

        # Determine sliding window for long input
        start_char = 0
        if self.cursor_pos >= max_chars:
            start_char = self.cursor_pos - max_chars + 1

        visible_text = self.input_buffer[start_char : start_char + max_chars]
        self._safe_addstr(max_y - 2, 0, "Input: " + visible_text)
        try:
            self.stdscr.clrtoeol()
            # Position cursor on the correct char in the UI
            cursor_ui_x = 7 + (self.cursor_pos - start_char)
            self.stdscr.move(max_y - 2, min(cursor_ui_x, max_x - 1))
        except curses.error:
            pass

    def show_footer_message(self, message: str):
        max_y, max_x = self.stdscr.getmaxyx()
        self._safe_addstr(max_y - 1, 0, message[: max_x - 1])
        try:
            self.stdscr.clrtoeol()
            self.stdscr.refresh()
        except curses.error:
            pass

    async def handle_input(self):
        """Handles user input."""
        key = self.stdscr.getch()
        if key == -1:
            return

        max_y, max_x = self.stdscr.getmaxyx()

        if key == 27:  # Esc key
            self.active_screen = 2 if self.active_screen == 1 else 1
            self.scroll_offset = 0
        elif key in (curses.KEY_BACKSPACE, 127, 8):  # Backspace variants
            if self.cursor_pos > 0:
                self.input_buffer = (
                    self.input_buffer[: self.cursor_pos - 1]
                    + self.input_buffer[self.cursor_pos :]
                )
                self.cursor_pos -= 1
        elif key == curses.KEY_DC:  # Delete key
            if self.cursor_pos < len(self.input_buffer):
                self.input_buffer = (
                    self.input_buffer[: self.cursor_pos]
                    + self.input_buffer[self.cursor_pos + 1 :]
                )
        elif key == curses.KEY_LEFT:
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor_pos = min(len(self.input_buffer), self.cursor_pos + 1)
        elif key == curses.KEY_HOME:
            self.cursor_pos = 0
        elif key == curses.KEY_END:
            self.cursor_pos = len(self.input_buffer)
        elif key == VOICE_CAPTURE_KEY_CODE:
            await self.handle_command("/voice-start")
        elif key == 4:  # Ctrl+D
            await self.handle_command("/quit")
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            if self.input_buffer.strip():
                # Add to history
                cmd_text = self.input_buffer.strip()
                if not self.input_history or self.input_history[-1] != cmd_text:
                    self.input_history.append(cmd_text)
                self.history_index = -1

                normalized = cmd_text.lower()
                if normalized in {"quit", "exit"}:
                    await self.handle_command("/quit")
                elif self.input_buffer.startswith("/"):
                    await self.handle_command(self.input_buffer)
                else:
                    self.interaction_queue.put_nowait(
                        {
                            "input": self.input_buffer,
                            "private_notes": "",
                            "audio_waveform": self.audio_waveform,
                        }
                    )
                    self.state["unprocessed_interactions"] += 1
                    await self.interaction_log_manager.append(
                        f"Input: {self.input_buffer}"
                    )
                self.input_buffer = ""
                self.cursor_pos = 0
        elif key == curses.KEY_UP:
            if self.input_history:
                if self.history_index == -1:
                    self.history_index = len(self.input_history) - 1
                else:
                    self.history_index = max(0, self.history_index - 1)
                self.input_buffer = self.input_history[self.history_index]
                self.cursor_pos = len(self.input_buffer)
        elif key == curses.KEY_DOWN:
            if self.history_index != -1:
                if self.history_index < len(self.input_history) - 1:
                    self.history_index += 1
                    self.input_buffer = self.input_history[self.history_index]
                else:
                    self.history_index = -1
                    self.input_buffer = ""
                self.cursor_pos = len(self.input_buffer)
        elif key == curses.KEY_PPAGE:  # Page Up
            max_log_length = (
                len(self.interaction_log_manager.interaction_log)
                if self.active_screen == 1
                else len(self.debug_log)
            )
            # Scroll half a screen or a full screen
            scroll_by = max_y - 5
            self.scroll_offset = min(self.scroll_offset + scroll_by, max_log_length - 1)
        elif key == curses.KEY_NPAGE:  # Page Down
            scroll_by = max_y - 5
            self.scroll_offset = max(0, self.scroll_offset - scroll_by)
        elif key == 9 and self.input_buffer.startswith(
            "/"
        ):  # Tab for command autocomplete
            self.autocomplete_command()
        elif 32 <= key <= 126:  # Printable characters
            self.input_buffer = (
                self.input_buffer[: self.cursor_pos]
                + chr(key)
                + self.input_buffer[self.cursor_pos :]
            )
            self.cursor_pos += 1

        self.render_input_line()
        try:
            self.stdscr.refresh()
        except curses.error:
            pass

    def autocomplete_command(self):
        prefix = self.input_buffer.strip().lower()
        matches = [cmd for cmd in self.COMMANDS if cmd.startswith(prefix)]
        if len(matches) == 1:
            self.input_buffer = matches[0]
        elif len(matches) > 1:
            self.show_footer_message("Matches: " + ", ".join(matches))
        else:
            self.show_footer_message("No command matches. Try /help")

    async def handle_command(self, command: str):
        raw = command.strip()
        await self.interaction_log_manager.append(f"Command: {raw}")
        cmd = raw.lower()
        parts = raw.split()
        model_manager = (
            getattr(self.interaction_processor, "llama_manager", None)
            if self.interaction_processor is not None
            else None
        )
        if cmd in {"/quit", "/exit"}:
            await self.interaction_log_manager.append("Shutting down...")
            self.show_footer_message("Shutting down...")
            await self._voice_loop.stop()
            self.state["voice_mode"] = "offline-disabled"
            self.state["voice_server_state"] = "stopped"
            self.state["voice_session_state"] = "disconnected"
            self.state["voice_activity_state"] = "idle"
            self.state["voice_last_event"] = "stopped"
            self._stop = True
            return
        if cmd == "/force-quit":
            await self.interaction_log_manager.append("Force quitting now...")
            self.show_footer_message("Force quitting now...")
            force_exit_now(130)
        if cmd == "/help":
            await self.interaction_log_manager.append(
                (
                    "Commands: /help, /model, /llm-status, /llm-diagnose, /smoke, /smoke-model, /smoke-all, "
                    "/voice-start, /voice-stop, /voice-status, /voice-diagnose, /voice-say <text>, "
                    "/voice-hear <text>, /voice-hear-file <path>, "
                    "/wake, /sleep, /quit, /force-quit. "
                    f"Voice start hotkey: {VOICE_CAPTURE_KEY_LABEL}. "
                    "Keys: Esc=toggle debug, Up/Down=history, PgUp/PgDn=scroll, Tab=autocomplete, Backspace=edit, Ctrl+D=quit."
                )
            )
            self.show_footer_message("Command help added to log.")
            return
        if cmd == "/model" or cmd.startswith("/model "):
            if model_manager is None:
                msg = "Model manager unavailable."
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            if len(parts) == 1:
                info = model_manager.get_model_info()
                aliases = ", ".join(model_manager.list_models().keys())
                msg = (
                    f"Model active: {info.get('alias')} ({info.get('backend_active')}); "
                    f"available: {aliases}. Use /model use <alias>."
                )
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            if len(parts) == 2 and parts[1].lower() in {"list", "ls"}:
                catalog = model_manager.list_models()
                details = ", ".join(
                    f"{name}({spec.get('backend')})" for name, spec in catalog.items()
                )
                msg = f"Available models: {details}"
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            if len(parts) == 3 and parts[1].lower() == "use":
                alias = parts[2]
                self.show_footer_message(f"Loading model '{alias}'...")
                try:
                    info = await asyncio.to_thread(model_manager.load_model, alias)
                    msg = (
                        f"Model switched to {info.get('alias')} "
                        f"({info.get('backend_active')})."
                    )
                except Exception as e:
                    msg = f"Model switch failed: {str(e)[:120]}"
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            usage = "Usage: /model | /model list | /model use <alias>"
            await self.interaction_log_manager.append(usage)
            self.show_footer_message(usage)
            return
        if cmd == "/llm-status":
            if model_manager is None:
                msg = "LLM status: DISABLED (deep reasoning bypassed)."
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return

            model_info = (
                model_manager.get_model_info() if model_manager is not None else {}
            )
            llm_busy = model_manager.is_busy() if model_manager is not None else False
            elapsed = ""
            started_at = self.state.get("last_processing_started_at")
            if self.state.get("is_processing") and started_at:
                try:
                    dt = datetime.fromisoformat(started_at)
                    elapsed = f"{(datetime.now() - dt).total_seconds():.1f}s"
                except Exception:
                    elapsed = "unknown"
            msg = (
                f"LLM status: alias={model_info.get('alias')} backend={model_info.get('backend_active')} "
                f"busy={llm_busy} "
                f"processing={self.state.get('is_processing')} "
                f"elapsed={elapsed} "
                f"unprocessed={self.state.get('unprocessed_interactions')} "
                f"last_status={self.state.get('last_processing_status', '')} "
                f"last_error={self.state.get('last_processing_error', '')}"
            )
            await self.interaction_log_manager.append(msg[:400])
            self.show_footer_message("LLM status added to log.")
            return
        if cmd == "/llm-diagnose":
            if model_manager is None:
                msg = "LLM diagnose: DISABLED (deep reasoning bypassed)."
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            if model_manager.is_busy():
                msg = "LLM diagnose: skipped (LLM busy with active request)."
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            self.show_footer_message("Running LLM diagnose...")
            try:
                lines = await asyncio.wait_for(
                    asyncio.to_thread(self._diagnose_llm_runtime, model_manager),
                    timeout=LLM_DIAG_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                lines = [f"LLM diagnose timeout after {LLM_DIAG_TIMEOUT_SECONDS}s"]
            for line in lines:
                await self.interaction_log_manager.append(line)
            self.show_footer_message("LLM diagnostics added to log.")
            return
        if cmd == "/voice-status":
            active = self._voice_loop.is_running
            p_manager = getattr(self._voice_loop, "personaplex_manager", None)
            p_status = p_manager.get_status() if p_manager else {}

            msg = (
                f"Voice loop: {'running' if active else 'stopped'} "
                f"mode={self.state.get('voice_mode')} "
                f"activity={self.state.get('voice_activity_state')} "
                f"loaded={p_status.get('loaded')} "
                f"opt={p_status.get('optimize_strategy')} "
                f"compile={p_status.get('torch_compile')} "
                f"graphs={p_status.get('cuda_graphs')} "
                f"dev={p_status.get('device')}"
            )
            self.state["voice_server_state"] = "running" if active else "stopped"
            self.state["voice_session_state"] = "local" if active else "disconnected"
            self.state["voice_activity_state"] = "listening" if active else "idle"
            self.state["voice_last_event"] = msg
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd == "/voice-test-tone":
            from utils import play_test_tone

            msg = "Playing 1s test tone (440Hz)..."
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            try:
                await asyncio.to_thread(play_test_tone)
                msg = "Test tone complete."
            except Exception as e:
                msg = f"Test tone failed: {e}"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd == "/voice-devices":
            try:
                import sounddevice as sd

                devices = str(sd.query_devices())
                for line in devices.split("\n"):
                    await self.interaction_log_manager.append(f"Audio Device: {line}")
                self.show_footer_message("Audio devices listed in log.")
            except Exception as e:
                msg = f"Failed to list audio devices: {e}"
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
            return
        if cmd == "/logic-reload":
            import importlib
            import functional_agent
            import interaction_processor
            import utils

            try:
                msg = "Hot-reloading logic modules..."
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)

                # 1. Stop current processor if it exists
                if self.interaction_processor:
                    self.interaction_processor.request_stop()
                    # We can't easily await the task here without a reference to it,
                    # but request_stop will make it exit its loop shortly.

                # 2. Reload modules
                importlib.reload(utils)
                importlib.reload(functional_agent)
                importlib.reload(interaction_processor)

                # 3. Re-instantiate
                from interaction_processor import InteractionProcessor
                from functional_agent import FunctionalAgent

                # Get the manager from the old processor
                p_manager = getattr(
                    self.interaction_processor, "personaplex_manager", None
                )
                llama_manager = getattr(
                    self.interaction_processor, "llama_manager", None
                )
                idx_manager = getattr(self.interaction_processor, "index_manager", None)

                new_processor = InteractionProcessor(
                    self.interaction_queue,
                    self.state,
                    llama_manager,
                    self.interaction_log_manager,
                    idx_manager,
                    voice_loop=self.voice_loop,
                    personaplex_manager=p_manager,
                )
                # Re-bind components
                new_processor.functional_agent = FunctionalAgent(
                    llama_manager, state=self.state
                )
                self.interaction_processor = new_processor

                # 4. Start the new processor task
                asyncio.create_task(new_processor.start())

                msg = "Logic and Utils modules reloaded and processor restarted."
            except Exception as e:
                msg = f"Logic reload failed: {e}"
                self.logger.exception("Logic reload failed")
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd.startswith("/voice-set-device "):
            parts = raw.split()
            if len(parts) < 3:
                msg = "Usage: /voice-set-device <in|out> <id>"
            else:
                target = parts[1].lower()
                try:
                    dev_id = int(parts[2])
                    from utils import set_audio_devices

                    if target == "in":
                        set_audio_devices(input_id=dev_id)
                    elif target == "out":
                        set_audio_devices(output_id=dev_id)
                    msg = f"Set {target} device to {dev_id}"
                except ValueError:
                    msg = "Invalid device ID"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd.startswith("/voice-say "):
            text = raw[11:].strip()
            if not text:
                return
            msg = f"Injecting voice response: {text}"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            # Use InteractionProcessor to handle the injection
            self.interaction_queue.put_nowait(
                {
                    "input": "[Direct Injection]",
                    "audio_waveform": None,
                    "override_response": text,
                }
            )
            return
        if cmd.startswith("/voice-hear "):
            text = raw[12:].strip()
            if not text:
                return
            msg = f"Injecting heard speech: {text}"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            pm = getattr(
                self.interaction_processor, "personaplex_manager", None
            ) or getattr(self._voice_loop, "personaplex_manager", None)
            vl = self._voice_loop
            if pm and vl:

                async def _do_hear():
                    stream = pm.hear_stream(text, PERSONAPLEX_VOICE_PROMPT)
                    await vl.say_stream(stream)

                asyncio.create_task(_do_hear())
            else:
                self.show_footer_message("/voice-hear requires PersonaPlex + VoiceLoop")
            return
        if cmd.startswith("/voice-hear-file "):
            path = raw[17:].strip()
            if not path:
                self.show_footer_message("/voice-hear-file requires a file path")
                return
            msg = f"Injecting audio file: {path}"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            pm = getattr(
                self.interaction_processor, "personaplex_manager", None
            ) or getattr(self._voice_loop, "personaplex_manager", None)
            vl = self._voice_loop
            if pm and vl:

                async def _do_hear_file():
                    stream = pm.hear_stream(
                        "", PERSONAPLEX_VOICE_PROMPT, user_wav_path=path
                    )
                    await vl.say_stream(stream)

                asyncio.create_task(_do_hear_file())
            else:
                self.show_footer_message(
                    "/voice-hear-file requires PersonaPlex + VoiceLoop"
                )
            return
        if cmd.startswith("/set-persona "):
            new_persona = raw[13:].strip()
            if not new_persona:
                return
            import config

            config.PERSONAPLEX_TEXT_PROMPT = new_persona
            msg = f"Persona updated: {new_persona[:50]}..."
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd.startswith("/voice-optimize "):
            mode = raw[16:].strip().lower()
            if mode not in ["auto", "eager", "compile", "graphs"]:
                msg = f"Invalid optimization mode: {mode}. Use auto, eager, compile, or graphs."
            else:
                import config

                config.PERSONAPLEX_OPTIMIZE = mode
                msg = f"Optimization strategy updated to: {mode}. (Applied on next inference)"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd == "/voice-start":
            active = self._voice_loop.is_running
            if active:
                msg = "Voice loop already running"
                await self._set_voice_state(
                    server="running",
                    session="local",
                    activity="listening",
                    event=msg,
                )
                self.state["voice_mode"] = "offline-continuous"
                return
            try:
                await self._set_voice_state(
                    server="starting",
                    session="disconnected",
                    activity="starting",
                    event="Starting offline continuous voice loop",
                )
                await self._voice_loop.start()
                self.state["voice_mode"] = "offline-continuous"
                await self._set_voice_state(
                    server="running",
                    session="local",
                    activity="listening",
                    event="Offline voice loop running in terminal",
                )
            except Exception as e:
                self.logger.error("Voice server start failed: %s", e)
                self.state["voice_mode"] = "server-error"
                await self._set_voice_state(
                    server="error",
                    session="disconnected",
                    activity="failed",
                    event=f"Voice start failed: {str(e)[:80]}",
                )
            return
        if cmd == "/voice-stop":
            await self._voice_loop.stop()
            self.state["voice_mode"] = "offline-disabled"
            await self._set_voice_state(
                server="stopped",
                session="disconnected",
                activity="idle",
                event="Voice loop stopped",
            )
            return
        if cmd == "/voice-diagnose":
            diag = await asyncio.to_thread(self._diagnose_voice_runtime)
            for line in diag:
                await self.interaction_log_manager.append(line)
            self.show_footer_message("Voice diagnostics added to log.")
            return
        if cmd == "/wake":
            self.state["manual_wake"] = True
            self.state["is_sleeping"] = False
            msg = "Agent manually woken from sleep."
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd == "/sleep":
            self.state["manual_wake"] = False
            msg = "Agent returned to autonomous sleep cycle."
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd in {"/smoke", "/smoke-all"}:
            self.show_footer_message("Running deterministic smoke test...")
            deterministic_result = await run_deterministic_smoke()
            deterministic_summary = summarize_smoke_result(deterministic_result)
            await self.interaction_log_manager.append(deterministic_summary)
            self.show_footer_message(deterministic_summary)
        if cmd in {"/smoke-model", "/smoke-all"}:
            if self.functional_agent is None:
                msg = "SMOKE model: FAIL functional agent unavailable"
                await self.interaction_log_manager.append(msg)
                self.show_footer_message(msg)
                return
            self.show_footer_message("Running model smoke test...")
            model_result = await run_model_smoke(self.functional_agent)
            model_summary = summarize_smoke_result(model_result)
            await self.interaction_log_manager.append(model_summary)
            self.show_footer_message(model_summary)
            return
        if cmd not in {
            "/model",
            "/llm-status",
            "/llm-diagnose",
            "/smoke",
            "/smoke-all",
            "/quit",
            "/exit",
            "/force-quit",
            "/voice-start",
            "/voice-stop",
            "/voice-status",
            "/voice-diagnose",
        }:
            msg = f"Unknown command: {command}. Try /help"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)

    def _diagnose_voice_runtime(self):
        lines = []
        py = _resolve_personaplex_python()
        lines.append(f"Voice diagnose: personaplex_python={py}")
        lines.append(f"Voice diagnose: python_exists={os.path.exists(py)}")

        check_script = (
            "results = []\n"
            "try:\n"
            "    import torch\n"
            "    cu = getattr(torch.version, 'cuda', 'N/A')\n"
            "    results.append(f'torch={torch.__version__} cuda={torch.cuda.is_available()} cu_ver={cu}')\n"
            "except ImportError:\n"
            "    results.append('torch=MISSING')\n"
            "try:\n"
            "    import moshi\n"
            "    results.append('moshi=OK')\n"
            "except ImportError:\n"
            "    results.append('moshi=MISSING')\n"
            "print(' '.join(results))\n"
        )

        command = [py, "-c", check_script]
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                lines.append(f"Voice diagnose: {result.stdout.strip()}")
            else:
                lines.append(
                    f"Voice diagnose: torch check failed rc={result.returncode}"
                )
                if result.stderr.strip():
                    lines.append(
                        f"Voice diagnose stderr: {result.stderr.strip()[:120]}"
                    )
        except Exception as e:
            lines.append(f"Voice diagnose: failed to run torch check: {str(e)[:120]}")
        return lines

    def _diagnose_llm_runtime(self, model_manager):
        lines = []
        info = model_manager.get_model_info()
        lines.append(
            f"LLM diagnose: alias={info.get('alias')} backend={info.get('backend_active')}"
        )
        test_prompt = "Reply with exactly: OK"
        try:
            result = model_manager.llm_call(test_prompt, 8)
            lines.append(f"LLM diagnose result: {str(result).strip()[:160]}")
        except Exception as e:
            lines.append(f"LLM diagnose error: {str(e)[:200]}")
        return lines

    def process_debug_queue(self):
        """Processes the debug queue."""
        while not self.debug_queue.empty():
            try:
                debug_message = self.debug_queue.get_nowait()
                self.debug_log.append(debug_message)
            except QueueEmpty:
                break
