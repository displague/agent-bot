import asyncio
import threading
import time
import random
import logging
from datetime import datetime, timedelta
from queue import Queue, Empty as QueueEmpty
import curses
import sys
import os
from threading import Lock
import json
from concurrent.futures import ThreadPoolExecutor
import contextlib
import torch
import sounddevice as sd
import numpy as np
import textwrap

from transformers import WhisperProcessor, WhisperModel

# Logging Setup
os.makedirs("logs", exist_ok=True)
os.makedirs("compressed_logs", exist_ok=True)
os.makedirs("index", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/application.log"),
    ],
)
logger = logging.getLogger("autonomous_system")

HARD_LOG_PATH = "logs/hard_log.jsonl"
COMPRESSED_LOG_PATH = "compressed_logs/compressed_log.jsonl"
INDEX_PATH = "index/context_index.json"
DAILY_SLEEP_START = 23
DAILY_SLEEP_END = 7

stderr_fileno = sys.stderr.fileno()
stderr_backup = os.dup(stderr_fileno)
original_stderr = sys.stderr

executor = ThreadPoolExecutor(max_workers=5)

###########################################################
# Utility & Feature extraction (placeholders)
###########################################################

def extract_text_features(text):
    return {}

def extract_audio_features(audio_waveform):
    return {}

def transcribe_audio(audio_waveform, sample_rate=16000):
    logger.debug("Transcribing audio...")
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperModel.from_pretrained("openai/whisper-small")
        model.eval()
        input_features = processor(audio_waveform, sampling_rate=sample_rate, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.decode(predicted_ids[0]).strip()
        logger.debug(f"Transcription result: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return ""

###########################################################
# Index Manager
###########################################################

class IndexManager:
    def __init__(self):
        self.index_path = INDEX_PATH

    def extract_keywords(self, text):
        return list(set(text.lower().split()))

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as index_file:
                return json.load(index_file)
        return {}

    def save_index(self, index):
        with open(self.index_path, "w") as index_file:
            json.dump(index, index_file)

    def index_interaction(self, entry):
        index = self.load_index()
        keywords = self.extract_keywords(entry.get("input", "") + " " + entry.get("output", ""))
        for keyword in keywords:
            if keyword in index:
                index[keyword].append(entry)
            else:
                index[keyword] = [entry]
        self.save_index(index)

    def search_context(self, keyword):
        index = self.load_index()
        return index.get(keyword.lower(), [])

###########################################################
# Interaction Log Manager
###########################################################

class InteractionLogManager:
    def __init__(self):
        self.interaction_log = []
        self.lock = asyncio.Lock()

    async def append(self, entry):
        async with self.lock:
            self.interaction_log.append(entry)

    async def get_display_log(self, max_items, scroll_offset=0):
        async with self.lock:
            return self.interaction_log[-(max_items + scroll_offset):(-scroll_offset if scroll_offset > 0 else None)]

###########################################################
# LlamaModelManager with Function Calling (Conceptual)
###########################################################

class LlamaModelManager:
    """
    This class manages LLM calls and now simulates "function calling" as per llama-3 function calling specs.
    It includes phases: planning, execution, digesting, validating, responding.
    We define a conceptual protocol for function calling here.
    """

    def __init__(self, model_path="model.bin"):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path)
        self.llm_lock = Lock()
        self.llm_context = []
        self.context_limit = 512

        # Define available functions (pseudo-code)
        # In practice, these would conform to Llama-3's function calling specification
        self.available_functions = {
            "search_index": self.fn_search_index,
            "schedule_event": self.fn_schedule_event
        }

    @contextlib.contextmanager
    def capture_llm_stderr(self):
        stderr_fd = sys.stderr.fileno()
        with open("logs/llm_stderr.log", "w") as f:
            old_stderr = os.dup(stderr_fd)
            os.dup2(f.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                sys.stderr = original_stderr

    def llm_call(self, prompt, max_tokens=512):
        with self.llm_lock, self.capture_llm_stderr():
            response = self.llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
            return response["choices"][0]["text"]

    def estimate_token_count(self, text):
        tokens = self.llm.tokenize(text.encode('utf-8'))
        return len(tokens)

    def update_context(self, new_entry):
        self.llm_context.append(new_entry)
        context_str = "\n".join(self.llm_context)
        total_tokens = self.estimate_token_count(context_str)
        while total_tokens > self.context_limit:
            self.llm_context.pop(0)
            context_str = "\n".join(self.llm_context)
            total_tokens = self.estimate_token_count(context_str)

    async def generate_private_notes(self, prompt):
        loop = asyncio.get_running_loop()
        analysis_prompt = f"""
Note any uncertainties or hesitations about the following prompt. Keep it concise.

Context:
{prompt}

Notes:"""
        try:
            private_notes = await loop.run_in_executor(executor, self.llm_call, analysis_prompt, 150)
            return private_notes.strip()
        except Exception as e:
            logger.error(f"Error generating private notes: {e}")
            return ""

    # Pseudo function calling mechanism:
    # The model might output something like:
    # {"name": "search_index", "arguments": {"keyword": "some keyword"}}
    # We'll parse that and call the corresponding Python function.
    def call_function(self, function_name, arguments):
        if function_name in self.available_functions:
            return self.available_functions[function_name](**arguments)
        else:
            return "Error: Function not found."

    def fn_search_index(self, keyword):
        # Placeholder - actual implementation might need a reference to index_manager
        # We'll just simulate
        return f"Searched for {keyword}, results: ..."

    def fn_schedule_event(self, event_type, message):
        # Placeholder - in reality would schedule with event_scheduler
        return f"Scheduled {event_type} event with message: {message}"

    async def run_phase(self, phase_name, prompt, notes):
        """
        Run a single phase by calling the LLM.
        If the LLM requests a function call, execute it and feed results back.
        """
        loop = asyncio.get_running_loop()

        # Prepare internal prompt for the phase
        internal_prompt = f"""
# Phase: {phase_name}
# Reflection:
{notes}
# Context:
{"\n".join(self.llm_context)}
# Instruction:
{prompt}

Now produce the {phase_name} result. If you need to call a function, output JSON in the format:
{{"name": "function_name", "arguments": {{...}}}}
Otherwise, produce the {phase_name} text directly.
"""
        response = await loop.run_in_executor(executor, self.llm_call, internal_prompt, 512)
        response = response.strip()

        # Check if response looks like a function call
        # Pseudo code: parse JSON if it matches function call pattern
        if response.startswith("{") and response.endswith("}"):
            # Attempt to parse function call
            try:
                call_data = json.loads(response)
                fname = call_data["name"]
                args = call_data["arguments"]
                function_result = self.call_function(fname, args)
                # Update context with function result
                self.update_context(f"Function Call: {fname}, args: {args}, result: {function_result}")
                return function_result
            except Exception as e:
                logger.error(f"Failed to parse function call: {e}")
                return "Error parsing function call."
        else:
            # Normal textual response
            self.update_context(f"{phase_name} Output: {response}")
            return response

###########################################################
# Event Scheduler
###########################################################

class EventScheduler:
    def __init__(self, state, interaction_log_manager, index_manager):
        self.event_queue = asyncio.Queue()
        self.state = state
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager

    async def start(self):
        logger.debug("Starting event scheduler")
        while True:
            event = await self.event_queue.get()
            asyncio.create_task(self.handle_event(event))
            await asyncio.sleep(1)

    async def schedule_event(self, event):
        logger.debug(f"Scheduling event: {event}")
        await self.event_queue.put(event)

    async def handle_event(self, event):
        event_type = event["type"]
        logger.info(f"Handling event: {event}")

        if event_type == "reminder":
            await self.interaction_log_manager.append(f"\nReminder: {event['message']}\n")
        elif event_type == "lookup":
            keyword = event["keyword"]
            results = self.index_manager.search_context(keyword)
            logger.debug(f"Lookup results: {results}")
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
            await self.interaction_log_manager.append(f"\nTraining: {event['message']}\n")

        self.state["next_event"] = "Not scheduled" if self.event_queue.empty() else "Event pending"

###########################################################
# TUI Renderer
###########################################################

class TUIRenderer:
    def __init__(self, stdscr, state, interaction_queue, interaction_log_manager):
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

    async def start(self):
        logger.debug("Starting TUI rendering...")
        curses.curs_set(1)
        self.stdscr.nodelay(True)
        while True:
            await self.render()
            await asyncio.sleep(0.1)

    def render_status_bar(self):
        max_y, max_x = self.stdscr.getmaxyx()
        sleep_status = "SLEEPING" if self.state["is_sleeping"] else "ACTIVE"
        listening_status = " LISTENING" if self.state.get("is_listening", False) else ""
        status_bar = (
            f" Status: {sleep_status}{listening_status} | Unprocessed: {self.state['unprocessed_interactions']} "
            f"| Thoughts: {self.state['ongoing_thoughts']} | Next event: {self.state['next_event']} "
        )
        self.stdscr.addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)
        self.stdscr.clrtoeol()

    async def render(self):
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
        display_log = await self.interaction_log_manager.get_display_log(max_log_lines, self.scroll_offset)
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
        max_y, max_x = self.stdscr.getmaxyx()
        current_y = 1
        max_log_lines = max_y - 4
        display_log = self.debug_log[-(max_log_lines + self.scroll_offset) : (-self.scroll_offset if self.scroll_offset > 0 else None)]
        for debug_message in display_log:
            wrapped_debug = textwrap.wrap(debug_message, max_x)
            for line in wrapped_debug:
                if current_y >= max_y - 3:
                    break
                self.stdscr.addstr(current_y, 0, line)
                current_y += 1

    def render_input_line(self):
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.addstr(
            max_y - 2, 0, "Input: " + self.input_buffer[: max_x - 7]
        )
        self.stdscr.clrtoeol()

    async def handle_input(self):
        key = self.stdscr.getch()
        if key == -1:
            return
        if key == 27:
            self.active_screen = 2 if self.active_screen == 1 else 1
        elif key == curses.KEY_BACKSPACE or key == 127:
            self.input_buffer = self.input_buffer[:-1]
        elif key == 22:  # Ctrl+V for voice input
            await self.handle_voice_input()
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.input_buffer.strip():
                self.interaction_queue.put(
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
        elif 32 <= key <= 126:
            self.input_buffer += chr(key)

        self.render_input_line()
        self.stdscr.refresh()

    async def handle_voice_input(self):
        max_y, max_x = self.stdscr.getmaxyx()
        self.state["is_listening"] = True
        self.render_status_bar()
        self.stdscr.refresh()
        audio_recorder = AudioRecorder()
        audio_recorder.start()
        self.stdscr.nodelay(False)
        self.stdscr.getch()
        self.stdscr.nodelay(True)
        sd.stop()
        audio_recorder.join()
        self.audio_waveform = audio_recorder.audio_waveform
        self.state["is_listening"] = False
        self.render_status_bar()
        self.stdscr.refresh()

        if self.audio_waveform is not None:
            transcribed_text = transcribe_audio(self.audio_waveform)
            if transcribed_text.strip():
                self.input_buffer = transcribed_text.strip()
            else:
                self.input_buffer = ""
                self.stdscr.addstr(max_y - 1, 0, "Transcription was empty.")
                self.stdscr.clrtoeol()
                self.stdscr.refresh()
                await asyncio.sleep(2)
        else:
            self.input_buffer = ""
            self.stdscr.addstr(max_y - 1, 0, "Recording failed.")
            self.stdscr.clrtoeol()
            self.stdscr.refresh()
            await asyncio.sleep(2)

    def process_debug_queue(self):
        while not self.debug_queue.empty():
            try:
                debug_message = self.debug_queue.get_nowait()
                self.debug_log.append(debug_message)
            except QueueEmpty:
                break

###########################################################
# Audio Recorder
###########################################################

class AudioRecorder(threading.Thread):
    def __init__(self, duration=5, sample_rate=16000):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_waveform = None

    def run(self):
        logger.debug("Recording audio...")
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
        )
        sd.wait()
        self.audio_waveform = torch.from_numpy(audio_data.flatten())
        logger.debug("Audio recording completed.")

###########################################################
# FunctionalAgent: Integrating the phases and function calls
###########################################################

class FunctionalAgent:
    """
    This agent uses the llama model manager to go through multiple phases
    for each request:
    1. Planning
    2. Execution (researching/solving)
    3. Digesting
    4. Validating
    5. Responding

    It uses function calls if needed.
    """
    def __init__(self, llama_manager: LlamaModelManager):
        self.llama_manager = llama_manager

    async def handle_request(self, prompt):
        # Generate private notes as a quick reflection
        private_notes = await self.llama_manager.generate_private_notes(prompt)

        # Phase 1: Planning
        plan_result = await self.llama_manager.run_phase("Planning", prompt, private_notes)

        # Phase 2: Execution (the model may request function calls or produce partial results)
        execution_result = await self.llama_manager.run_phase("Execution", prompt, plan_result)

        # Phase 3: Digesting results
        digest_result = await self.llama_manager.run_phase("Digesting", prompt, execution_result)

        # Phase 4: Validating correctness
        validate_result = await self.llama_manager.run_phase("Validating", prompt, digest_result)

        # Phase 5: Final response to user
        final_response = await self.llama_manager.run_phase("Responding", prompt, validate_result)

        return final_response

###########################################################
# Interaction Processor & Thought Generator
###########################################################

class InteractionProcessor:
    """
    Now uses the FunctionalAgent to handle each interaction in phases.
    """
    def __init__(self, interaction_queue, state, llama_manager, interaction_log_manager, index_manager):
        self.interaction_queue = interaction_queue
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager
        self.functional_agent = FunctionalAgent(self.llama_manager)

    async def start(self):
        logger.debug("Starting interaction processing...")
        while True:
            try:
                interaction = self.interaction_queue.get_nowait()
                user_input = interaction.get("input", "")
                audio_waveform = interaction.get("audio_waveform", None)

                logger.info(f"Processing interaction: {user_input}")
                if audio_waveform is not None:
                    audio_features = extract_audio_features(audio_waveform)

                # Process request with multi-phase approach
                response = await self.functional_agent.handle_request(user_input)
                logger.info(f"Response: {response}")

                await self.interaction_log_manager.append(f"Thought: {response}")
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input": user_input,
                    "output": response,
                }
                with open(HARD_LOG_PATH, "a") as log_file:
                    log_file.write(json.dumps(log_entry) + "\n")

                self.index_manager.index_interaction(log_entry)
                self.state["unprocessed_interactions"] = max(0, self.state["unprocessed_interactions"] - 1)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            except QueueEmpty:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in process_interactions: {e}")
                await asyncio.sleep(1)

class ThoughtGenerator:
    """
    Generates autonomous thoughts periodically, also using phases.
    """
    def __init__(self, state, llama_manager, interaction_log_manager, event_scheduler):
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.event_scheduler = event_scheduler
        self.functional_agent = FunctionalAgent(self.llama_manager)

    async def start(self):
        logger.debug("Starting autonomous thought generation...")
        while True:
            current_hour = datetime.now().hour
            if DAILY_SLEEP_START <= current_hour or current_hour < DAILY_SLEEP_END:
                self.state["is_sleeping"] = True
                await asyncio.sleep(random.uniform(5, 10))
            else:
                self.state["is_sleeping"] = False
                logger.debug("Generating new autonomous thoughts")
                tasks = [asyncio.create_task(self.generate_thought()) for _ in range(3)]
                await asyncio.gather(*tasks)
                await asyncio.sleep(random.uniform(1, 3))

    async def generate_thought(self):
        self.state["ongoing_thoughts"] += 1
        thought_prompt = f"Autonomous thought at {datetime.now().strftime('%H:%M:%S')}"
        try:
            response = await self.functional_agent.handle_request(thought_prompt)
            self.state.setdefault("current_thoughts", []).append(response)
            await self.interaction_log_manager.append(f"Thought: {response}")
            await asyncio.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}")
            await asyncio.sleep(1)
        finally:
            self.state["ongoing_thoughts"] = max(0, self.state["ongoing_thoughts"] - 1)
            if "current_thoughts" in self.state and response in self.state["current_thoughts"]:
                self.state["current_thoughts"].remove(response)

###########################################################
# EventCompressor
###########################################################

class EventCompressor:
    def __init__(self, llama_manager, event_scheduler):
        self.llama_manager = llama_manager
        self.event_scheduler = event_scheduler

    async def start(self):
        logger.debug("Starting periodic event compression...")
        while True:
            await self.compress_events()
            await asyncio.sleep(3600)

    async def compress_events(self):
        logger.debug("Starting event compression")
        if not os.path.exists(HARD_LOG_PATH):
            logger.debug("No logs to compress.")
            return
        try:
            with open(HARD_LOG_PATH, "r") as log_file:
                logs = [json.loads(line) for line in log_file if line.strip()]
            if not logs:
                logger.debug("No events to compress.")
                return

            events_text = ""
            for entry in logs:
                input_text = entry.get('input', '')
                output_text = entry.get('output', '')
                events_text += f"Task: {input_text}\nThought: {output_text}\n"

            if not events_text.strip():
                logger.debug("No events to compress.")
                return

            # Could also use function-calling approach if needed
            prompt = f"""Summarize the following interactions for future reference:

{events_text}

Summary:"""
            summary = await asyncio.get_event_loop().run_in_executor(
                executor, self.llama_manager.llm_call, prompt
            )
            summary = summary.strip()
            compressed_entry = {"timestamp": datetime.now().isoformat(), "summary": summary}
            with open(COMPRESSED_LOG_PATH, "a") as comp_log_file:
                comp_log_file.write(json.dumps(compressed_entry) + "\n")
            logger.info("Event compression completed")

            rag_event = {"type": "rag_completed", "trigger_time": datetime.now()}
            await self.event_scheduler.schedule_event(rag_event)
        except Exception as e:
            logger.error(f"Error in compress_events: {e}")
            await asyncio.sleep(1)

###########################################################
# Main Application
###########################################################

class MainApplication:
    def __init__(self):
        self.state = {
            "unprocessed_interactions": 0,
            "ongoing_thoughts": 0,
            "next_event": "Not scheduled",
            "is_sleeping": False,
        }
        self.interaction_queue = Queue()
        self.llama_manager = LlamaModelManager()
        self.interaction_log_manager = InteractionLogManager()
        self.index_manager = IndexManager()
        self.event_scheduler = EventScheduler(self.state, self.interaction_log_manager, self.index_manager)
        self.interaction_processor = InteractionProcessor(self.interaction_queue, self.state, self.llama_manager, self.interaction_log_manager, self.index_manager)
        self.thought_generator = ThoughtGenerator(self.state, self.llama_manager, self.interaction_log_manager, self.event_scheduler)
        self.event_compressor = EventCompressor(self.llama_manager, self.event_scheduler)

    async def main(self, stdscr):
        loop = asyncio.get_running_loop()
        logger.debug("Starting main event loop...")
        self.tui_renderer = TUIRenderer(stdscr, self.state, self.interaction_queue, self.interaction_log_manager)
        tasks = [
            asyncio.create_task(self.tui_renderer.start()),
            asyncio.create_task(self.interaction_processor.start()),
            asyncio.create_task(self.thought_generator.start()),
            asyncio.create_task(self.event_scheduler.start()),
            asyncio.create_task(self.event_compressor.start()),
        ]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Exception in main: {e}")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        logger.debug("Launching TUI application...")
        app = MainApplication()
        curses.wrapper(lambda stdscr: asyncio.run(app.main(stdscr)))
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt received, shutting down...")
        os.dup2(stderr_backup, stderr_fileno)
        print("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        sys.stderr = original_stderr
        pending = asyncio.all_tasks(loop=asyncio.get_running_loop())
        for task in pending:
            task.cancel()
        try:
            asyncio.get_running_loop().run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        except Exception as e:
            logger.error(f"Error while cancelling tasks: {e}")
        os.dup2(stderr_backup, stderr_fileno)
        sys.exit(0)
