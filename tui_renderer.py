# tui_renderer.py

import asyncio
import curses
import os
import tempfile
import textwrap
from queue import Queue, Empty as QueueEmpty
import logging

from config import (
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    VOICE_CAPTURE_SECONDS,
    VOICE_SAMPLE_RATE,
)
from utils import capture_microphone_to_wav, play_wav_file, run_personaplex_offline

logger = logging.getLogger("autonomous_system.tui_renderer")


class TUIRenderer:
    def __init__(self, stdscr, state, interaction_queue, interaction_log_manager):
        self.logger = logging.getLogger("autonomous_system.tui_renderer")
        self.stdscr = stdscr
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self.debug_queue = Queue()
        self.active_screen = 1
        self.input_buffer = ""
        self.debug_log = []
        self.scroll_offset = 0
        self.audio_waveform = None
        self.state.setdefault("is_listening", False)
        self.state.setdefault("voice_prompt", PERSONAPLEX_VOICE_PROMPT)
        self.state.setdefault("persona_prompt", PERSONAPLEX_TEXT_PROMPT)

    async def start(self):
        """Starts the TUI rendering."""
        self.logger.debug("Starting TUI rendering...")
        curses.curs_set(1)
        self.stdscr.nodelay(True)
        while True:
            await self.render()
            await asyncio.sleep(0.1)

    def render_status_bar(self):
        """Renders the status bar."""
        max_y, max_x = self.stdscr.getmaxyx()
        sleep_status = "SLEEPING" if self.state["is_sleeping"] else "ACTIVE"
        listening_status = " LISTENING" if self.state.get("is_listening", False) else ""
        status_bar = (
            f" Status: {sleep_status}{listening_status} | Unprocessed: {self.state['unprocessed_interactions']} "
            f"| Thoughts: {self.state['ongoing_thoughts']} | Next event: {self.state['next_event']} "
            f"| Voice: {self.state.get('voice_prompt', PERSONAPLEX_VOICE_PROMPT)} "
        )
        self.stdscr.addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)
        self.stdscr.clrtoeol()

    async def render(self):
        """Renders the main screen or debug screen."""
        self.stdscr.clear()
        self.render_status_bar()

        if self.active_screen == 1:
            await self.render_main_screen()
        elif self.active_screen == 2:
            self.render_debug_screen()

        self.render_input_line()
        self.stdscr.refresh()

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
                self.stdscr.addstr(current_y, 0, line)
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
                self.stdscr.addstr(current_y, 0, line)
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
                self.stdscr.addstr(current_y, 0, line)
                current_y += 1

    def render_input_line(self):
        """Renders the input line."""
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.addstr(max_y - 2, 0, "Input: " + self.input_buffer[: max_x - 7])
        self.stdscr.clrtoeol()

    async def handle_input(self):
        """Handles user input."""
        key = self.stdscr.getch()
        if key == -1:
            return
        if key == 27:  # Esc key
            self.active_screen = 2 if self.active_screen == 1 else 1
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace or delete
            self.input_buffer = self.input_buffer[:-1]
        elif key == 22:  # Ctrl+V for voice input
            await self.handle_voice_input()
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            if self.input_buffer.strip():
                self.interaction_queue.put_nowait(
                    {
                        "input": self.input_buffer,
                        "private_notes": "",
                        "audio_waveform": self.audio_waveform,
                    }
                )
                self.state["unprocessed_interactions"] += 1
                await self.interaction_log_manager.append(f"Input: {self.input_buffer}")
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
        elif 32 <= key <= 126:  # Printable characters
            self.input_buffer += chr(key)

        self.render_input_line()
        self.stdscr.refresh()

    async def handle_voice_input(self):
        """Handles voice input using PersonaPlex."""
        max_y, max_x = self.stdscr.getmaxyx()
        try:
            self.state["is_listening"] = True
            self.render_status_bar()
            self.stdscr.addstr(
                max_y - 1,
                0,
                f"Recording voice for {VOICE_CAPTURE_SECONDS}s...",
            )
            self.stdscr.clrtoeol()
            self.stdscr.refresh()

            with tempfile.TemporaryDirectory(prefix="agentbot_voice_") as tmpdir:
                input_wav = os.path.join(tmpdir, "input.wav")
                output_wav = os.path.join(tmpdir, "output.wav")
                output_text = os.path.join(tmpdir, "output_text.json")

                await asyncio.to_thread(
                    capture_microphone_to_wav,
                    input_wav,
                    VOICE_CAPTURE_SECONDS,
                    VOICE_SAMPLE_RATE,
                )
                self.stdscr.addstr(max_y - 1, 0, "Processing PersonaPlex response...")
                self.stdscr.clrtoeol()
                self.stdscr.refresh()

                response = await asyncio.to_thread(
                    run_personaplex_offline,
                    input_wav,
                    output_wav,
                    output_text=output_text,
                    voice_prompt=self.state.get(
                        "voice_prompt", PERSONAPLEX_VOICE_PROMPT
                    ),
                    text_prompt=self.state.get(
                        "persona_prompt", PERSONAPLEX_TEXT_PROMPT
                    ),
                )
                await asyncio.to_thread(play_wav_file, output_wav)

                generated_text = response.get("generated_text", "").strip()
                if generated_text:
                    await self.interaction_log_manager.append(
                        f"Voice Response: {generated_text}"
                    )
                else:
                    await self.interaction_log_manager.append(
                        "Voice Response: [audio generated]"
                    )

                self.stdscr.addstr(max_y - 1, 0, "Voice turn completed.")
                self.stdscr.clrtoeol()
        except Exception as e:
            self.logger.error(f"Voice input failed: {e}")
            self.stdscr.addstr(max_y - 1, 0, f"Voice error: {str(e)[: max_x - 13]}")
            self.stdscr.clrtoeol()
        finally:
            self.state["is_listening"] = False
            self.render_status_bar()
            self.stdscr.refresh()

    def process_debug_queue(self):
        """Processes the debug queue."""
        while not self.debug_queue.empty():
            try:
                debug_message = self.debug_queue.get_nowait()
                self.debug_log.append(debug_message)
            except QueueEmpty:
                break
