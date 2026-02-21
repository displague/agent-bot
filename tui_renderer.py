# tui_renderer.py

import asyncio
import curses
import textwrap
from queue import Queue, Empty as QueueEmpty
import logging

from config import (
    PERSONAPLEX_SERVER_URL,
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    VOICE_AUTO_START_ON_LAUNCH,
    VOICE_CAPTURE_KEY_CODE,
    VOICE_CAPTURE_KEY_LABEL,
)
from smoke_test_runner import (
    run_deterministic_smoke,
    run_model_smoke,
    summarize_smoke_result,
)
from utils import start_personaplex_server, stop_personaplex_server

logger = logging.getLogger("autonomous_system.tui_renderer")


class TUIRenderer:
    COMMANDS = [
        "/help",
        "/smoke",
        "/smoke-model",
        "/smoke-all",
        "/voice-start",
        "/voice-stop",
        "/voice-status",
        "/quit",
        "/exit",
    ]

    def __init__(
        self,
        stdscr,
        state,
        interaction_queue,
        interaction_log_manager,
        functional_agent=None,
    ):
        self.logger = logging.getLogger("autonomous_system.tui_renderer")
        self.stdscr = stdscr
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self.functional_agent = functional_agent
        self.debug_queue = Queue()
        self.active_screen = 1
        self.input_buffer = ""
        self.debug_log = []
        self.scroll_offset = 0
        self.audio_waveform = None
        self._stop = False
        self._refresh_interval_seconds = 0.2
        self._voice_server_handle = None
        self._pending_voice_event = None
        self.state.setdefault("is_listening", False)
        self.state.setdefault("voice_prompt", PERSONAPLEX_VOICE_PROMPT)
        self.state.setdefault("persona_prompt", PERSONAPLEX_TEXT_PROMPT)
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
            f"Type /help for commands. {VOICE_CAPTURE_KEY_LABEL} starts voice server mode. Esc toggles debug."
        )
        if VOICE_AUTO_START_ON_LAUNCH:
            await self._set_voice_state(
                server="starting",
                session="disconnected",
                activity="starting",
                event="Auto-starting voice server at launch",
            )
            await self.handle_command("/voice-start")
        try:
            while not self._stop:
                try:
                    self._refresh_voice_health()
                    if self._pending_voice_event:
                        event = self._pending_voice_event
                        self._pending_voice_event = None
                        await self.interaction_log_manager.append(f"Voice: {event}")
                        self.show_footer_message(f"Voice: {event}")
                    await self.render()
                except Exception as e:
                    self.logger.exception("TUI render loop error: %s", e)
                    self.show_footer_message(f"TUI error: {str(e)[:120]}")
                    await self.interaction_log_manager.append(f"TUI error: {e}")
                await asyncio.sleep(self._refresh_interval_seconds)
        finally:
            stop_personaplex_server(self._voice_server_handle)
            self._voice_server_handle = None

    async def _set_voice_state(self, *, server=None, session=None, activity=None, event=None):
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

    def _refresh_voice_health(self):
        if self._voice_server_handle is None:
            return
        if self._voice_server_handle.process.poll() is not None:
            self._voice_server_handle = None
            self.state["voice_mode"] = "offline-disabled"
            self.state["voice_server_state"] = "stopped"
            self.state["voice_session_state"] = "disconnected"
            self.state["voice_activity_state"] = "failed"
            self.state["voice_last_event"] = "server exited unexpectedly"
            self._pending_voice_event = "server exited unexpectedly"

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
        processing_status = " PROCESSING" if self.state.get("is_processing", False) else ""
        voice_mode = self.state.get("voice_mode", "unknown")
        voice_server_state = self.state.get("voice_server_state", "unknown")
        voice_session_state = self.state.get("voice_session_state", "unknown")
        voice_activity_state = self.state.get("voice_activity_state", "unknown")
        voice_last_event = self.state.get("voice_last_event", "none")
        status_bar = (
            f" Status: {sleep_status}{listening_status}{processing_status} | Unprocessed: {self.state['unprocessed_interactions']} "
            f"| Thoughts: {self.state['ongoing_thoughts']} | Next event: {self.state['next_event']} "
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
        """Renders the main screen."""
        max_y, max_x = self.stdscr.getmaxyx()
        current_y = 1
        current_thoughts = self.state.get("current_thoughts", [])
        for thought in current_thoughts:
            wrapped_thought = textwrap.wrap(f"Thought: {thought}", max_x)
            for line in wrapped_thought:
                if current_y >= max_y - 3:
                    break
                self._safe_addstr(current_y, 0, line)
                current_y += 1

        max_log_lines = max_y - current_y - 3
        display_log = await self.interaction_log_manager.get_display_log(
            max_log_lines, self.scroll_offset
        )
        for interaction in display_log:
            if current_y >= max_y - 3:
                break
            wrapped_interaction = textwrap.wrap(interaction, max_x)
            for line in wrapped_interaction:
                if current_y >= max_y - 3:
                    break
                self._safe_addstr(current_y, 0, line)
                current_y += 1

    def render_debug_screen(self):
        """Renders the debug screen."""
        max_y, max_x = self.stdscr.getmaxyx()
        current_y = 1
        max_log_lines = max_y - 4
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
        """Renders the input line."""
        max_y, max_x = self.stdscr.getmaxyx()
        self._safe_addstr(max_y - 2, 0, "Input: " + self.input_buffer[: max_x - 7])
        try:
            self.stdscr.clrtoeol()
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
        if key == 27:  # Esc key
            self.active_screen = 2 if self.active_screen == 1 else 1
        elif key in (curses.KEY_BACKSPACE, 127, 8):  # Backspace variants
            self.input_buffer = self.input_buffer[:-1]
        elif key == VOICE_CAPTURE_KEY_CODE:
            await self.handle_command("/voice-start")
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            if self.input_buffer.strip():
                normalized = self.input_buffer.strip().lower()
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
        elif key == curses.KEY_UP:
            max_log_length = (
                len(self.interaction_log_manager.interaction_log)
                if self.active_screen == 1
                else len(self.debug_log)
            )
            self.scroll_offset = min(self.scroll_offset + 1, max_log_length)
        elif key == curses.KEY_DOWN:
            self.scroll_offset = max(self.scroll_offset - 1, 0)
        elif key == 9 and self.input_buffer.startswith("/"):  # Tab for command autocomplete
            self.autocomplete_command()
        elif 32 <= key <= 126:  # Printable characters
            self.input_buffer += chr(key)

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
        cmd = command.strip().lower()
        if cmd in {"/quit", "/exit"}:
            await self.interaction_log_manager.append("Shutting down...")
            self.show_footer_message("Shutting down...")
            stop_personaplex_server(self._voice_server_handle)
            self._voice_server_handle = None
            self.state["voice_mode"] = "offline-disabled"
            self.state["voice_server_state"] = "stopped"
            self.state["voice_session_state"] = "disconnected"
            self.state["voice_activity_state"] = "idle"
            self.state["voice_last_event"] = "stopped"
            self._stop = True
            return
        if cmd == "/help":
            await self.interaction_log_manager.append(
                (
                    "Commands: /help, /smoke, /smoke-model, /smoke-all, "
                    "/voice-start, /voice-stop, /voice-status, /quit. "
                    f"Voice start hotkey: {VOICE_CAPTURE_KEY_LABEL}. "
                    "Keys: Esc=toggle debug, Up/Down=scroll, Tab=autocomplete, Backspace=edit."
                )
            )
            self.show_footer_message("Command help added to log.")
            return
        if cmd == "/voice-status":
            active = (
                self._voice_server_handle is not None
                and self._voice_server_handle.process.poll() is None
            )
            log_hint = (
                f" log={self._voice_server_handle.log_path}"
                if active and self._voice_server_handle is not None
                else ""
            )
            msg = (
                f"Voice server: {'running' if active else 'stopped'} "
                f"({PERSONAPLEX_SERVER_URL}){log_hint}"
            )
            self.state["voice_server_state"] = "running" if active else "stopped"
            self.state["voice_session_state"] = "unknown" if active else "disconnected"
            self.state["voice_activity_state"] = "waiting-client" if active else "idle"
            self.state["voice_last_event"] = msg
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)
            return
        if cmd == "/voice-start":
            active = (
                self._voice_server_handle is not None
                and self._voice_server_handle.process.poll() is None
            )
            if active:
                msg = f"Voice server already running at {PERSONAPLEX_SERVER_URL}"
                await self._set_voice_state(
                    server="running",
                    session="unknown",
                    activity="waiting-client",
                    event=msg,
                )
                self.state["voice_mode"] = "server"
                return
            try:
                await self._set_voice_state(
                    server="starting",
                    session="disconnected",
                    activity="starting",
                    event="Starting PersonaPlex voice server",
                )
                self._voice_server_handle = await asyncio.to_thread(start_personaplex_server)
                self.state["voice_mode"] = "server"
                await self._set_voice_state(
                    server="running",
                    session="unknown",
                    activity="waiting-client",
                    event=(
                        "Server running. Open PersonaPlex UI to begin stream: "
                        f"{PERSONAPLEX_SERVER_URL}"
                    ),
                )
            except Exception as e:
                self.logger.error("Voice server start failed: %s", e)
                self.state["voice_mode"] = "server-error"
                await self._set_voice_state(
                    server="error",
                    session="disconnected",
                    activity="failed",
                    event=f"Server start failed: {str(e)[:80]}",
                )
            return
        if cmd == "/voice-stop":
            stop_personaplex_server(self._voice_server_handle)
            self._voice_server_handle = None
            self.state["voice_mode"] = "offline-disabled"
            await self._set_voice_state(
                server="stopped",
                session="disconnected",
                activity="idle",
                event="Voice server stopped",
            )
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
            "/smoke",
            "/smoke-all",
            "/quit",
            "/exit",
            "/voice-start",
            "/voice-stop",
            "/voice-status",
        }:
            msg = f"Unknown command: {command}. Try /help"
            await self.interaction_log_manager.append(msg)
            self.show_footer_message(msg)

    def process_debug_queue(self):
        """Processes the debug queue."""
        while not self.debug_queue.empty():
            try:
                debug_message = self.debug_queue.get_nowait()
                self.debug_log.append(debug_message)
            except QueueEmpty:
                break
