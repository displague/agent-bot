import asyncio
import threading
import time
import random
import logging
from datetime import datetime, timedelta
from queue import Queue, Empty as QueueEmpty
import curses
import requests
from llama_cpp import Llama
import sys
import os
from threading import Lock
import json
from concurrent.futures import ThreadPoolExecutor
import contextlib
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel

# Set up logging for debug
logging.basicConfig(
    level=logging.DEBUG,  # Adjusted logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler(sys.stdout),  # Ensure logs go to stdout
    ],
)
logger = logging.getLogger("autonomous_system")

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("compressed_logs", exist_ok=True)
os.makedirs("index", exist_ok=True)

# Paths to files
HARD_LOG_PATH = "logs/hard_log.jsonl"
COMPRESSED_LOG_PATH = "compressed_logs/compressed_log.jsonl"
EVENT_QUEUE_PATH = "logs/event_queue.jsonl"
INDEX_PATH = "index/context_index.json"

# Redirect stderr to capture llama_cpp output during model calls
stderr_fileno = sys.stderr.fileno()
stderr_backup = os.dup(stderr_fileno)
original_stderr = sys.stderr

# Constants
daily_sleep_start = 23  # 11 PM
daily_sleep_end = 7  # 7 AM

# Create an executor for running blocking LLM calls
executor = ThreadPoolExecutor(max_workers=5)

class LlamaModelManager:
    def __init__(self, model_path="model.bin"):
        logger.debug("Initializing Llama model...")
        self.llm = Llama(model_path=model_path)
        self.llm_lock = Lock()
        self.llm_context = ""

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

    def llm_call(self, prompt):
        with self.llm_lock, self.capture_llm_stderr():
            response = self.llm(prompt, max_tokens=512, stop=["\n\n"])
            return response["choices"][0]["text"]

    def llm_call_private_notes(self, prompt):
        with self.llm_lock, self.capture_llm_stderr():
            response = self.llm(prompt, max_tokens=150, stop=["\n\n"])
            return response["choices"][0]["text"]

    async def generate_llama_response(self, prompt, notes):
        logger.debug(f"Generating response for prompt: {prompt}")
        loop = asyncio.get_running_loop()
        try:
            internal_prompt = f"""
# Reflection:
{notes}

# Conversation:
{self.llm_context}
Task: {prompt}
Thought:"""
            response = await loop.run_in_executor(executor, self.llm_call, internal_prompt)
            generated_text = response.strip()
            # Update context, ensuring clear separation between inputs and thoughts
            self.llm_context += f"\nTask: {prompt}\nThought: {generated_text}"
            logger.debug(f"Generated response: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response from Llama model: {e}")
            await asyncio.sleep(1)
            return "Error generating response."

    async def generate_private_notes(self, prompt):
        logger.debug("Generating private notes")
        loop = asyncio.get_running_loop()
        try:
            analysis_prompt = f"""
Note any uncertainties or hesitations regarding the following prompt. Keep it concise and relevant.

Context Information:
{prompt}

Notes (max 30 words):"""
            private_notes = await loop.run_in_executor(
                executor, self.llm_call_private_notes, analysis_prompt
            )
            private_notes = private_notes.strip()
            logger.debug(f"Private Notes: {private_notes}")
            return private_notes
        except Exception as e:
            logger.error(f"Error generating private notes: {e}")
            await asyncio.sleep(1)
            return ""

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

class IndexManager:
    def __init__(self):
        self.index_path = INDEX_PATH

    def extract_keywords(self, text):
        # Simple keyword extraction (could use NLP techniques)
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
        logger.debug("Indexing interaction")
        index = self.load_index()
        keywords = self.extract_keywords(entry["input"] + " " + entry["output"])
        for keyword in keywords:
            if keyword in index:
                index[keyword].append(entry)
            else:
                index[keyword] = [entry]
        self.save_index(index)

    def search_context(self, keyword):
        index = self.load_index()
        return index.get(keyword.lower(), [])

class EventScheduler:
    def __init__(self, state, interaction_log_manager, index_manager):
        self.event_queue = asyncio.Queue()
        self.state = state
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager

    async def start(self):
        logger.debug("Starting event scheduler")
        while True:
            try:
                event = await self.event_queue.get()
                asyncio.create_task(self.handle_event(event))
            except Exception as e:
                logger.error(f"Error in event scheduler: {e}")
            await asyncio.sleep(1)

    async def schedule_event(self, event):
        logger.debug(f"Scheduling event: {event}")
        await self.event_queue.put(event)

    async def handle_event(self, event):
        event_type = event["type"]
        logger.info(f"Handling event: {event}")
        if event_type == "reminder":
            message = event["message"]
            logger.debug(f"Reminder: {message}")
            await self.interaction_log_manager.append(f"\nReminder: {message}\n")
        elif event_type == "lookup":
            keyword = event["keyword"]
            results = self.index_manager.search_context(keyword)
            notes = f"Lookup results for '{keyword}': {results}"
            logger.debug(f"Notes after lookup: {notes}")
        elif event_type == "deferred_topic":
            topic = event["topic"]
            logger.debug(f"Revisiting deferred topic: {topic}")
            message = f"I've thought more about {topic} and would like to discuss it further."
            await self.interaction_log_manager.append(f"\nThought: {message}\n")
        elif event_type == "rag_completed":
            logger.debug("Processing RAG completed event")
            message = "RAG processing has completed. Proceeding with training adjustments."
            training_event = {
                "type": "training",
                "message": message,
                "trigger_time": datetime.now() + timedelta(minutes=5),
            }
            await self.schedule_event(training_event)
        elif event_type == "training":
            message = event["message"]
            logger.debug(f"Training: {message}")
            await self.interaction_log_manager.append(f"\nTraining: {message}\n")
        self.state["next_event"] = "Not scheduled" if self.event_queue.empty() else "Event pending"

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

    async def start(self):
        logger.debug("Starting TUI rendering...")
        curses.curs_set(1)
        self.stdscr.nodelay(True)
        while True:
            await self.render()
            await asyncio.sleep(0.1)

    async def render(self):
        self.stdscr.clear()
        max_y, max_x = self.stdscr.getmaxyx()

        if self.active_screen == 1:
            sleep_status = "SLEEPING" if self.state["is_sleeping"] else "ACTIVE"
            status_bar = f" Status: {sleep_status} | Unprocessed: {self.state['unprocessed_interactions']} | Thoughts: {self.state['ongoing_thoughts']} | Next event: {self.state['next_event']} "
            self.stdscr.addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)

            current_y = 1
            # Display current thoughts
            current_thoughts = self.state.get("current_thoughts", [])
            for thought in current_thoughts:
                self.stdscr.addstr(current_y, 0, f"Thought: {thought[:max_x]}")
                current_y += 1

            # Display interaction log
            display_log = await self.interaction_log_manager.get_display_log(max_y - current_y - 2, self.scroll_offset)
            for interaction in display_log:
                if current_y >= max_y - 2:
                    break
                self.stdscr.addstr(current_y, 0, interaction[:max_x])
                current_y += 1
        elif self.active_screen == 2:
            # Display debug log
            current_y = 1
            display_log = self.debug_log[-(max_y - 4 + self.scroll_offset):(-self.scroll_offset if self.scroll_offset > 0 else None)]
            for debug_message in display_log:
                self.stdscr.addstr(current_y, 0, debug_message[:max_x])
                current_y += 1

        self.stdscr.addstr(max_y - 2, 0, "Input: " + self.input_buffer[: max_x - 5])
        self.stdscr.refresh()

        key = self.stdscr.getch()
        if key == 27:
            logger.debug("Switching screen view")
            self.active_screen = 2 if self.active_screen == 1 else 1
        elif key == curses.KEY_BACKSPACE or key == 127:
            self.input_buffer = self.input_buffer[:-1]
        elif key == ord("v"):  # 'v' key for voice input
            self.stdscr.addstr(max_y - 1, 0, "Listening... (Press any key to stop)")
            self.stdscr.refresh()
            audio_recorder = AudioRecorder(duration=5)
            audio_recorder.start()
            self.stdscr.getch()  # Wait for any key press
            sd.stop()  # Stop recording when any key is pressed
            audio_recorder.join()
            self.audio_waveform = audio_recorder.audio_waveform
            if self.audio_waveform is not None:
                transcribed_text = transcribe_audio(self.audio_waveform)
                if transcribed_text.strip():
                    self.input_buffer = transcribed_text
                else:
                    self.input_buffer = ""
            else:
                self.input_buffer = ""
            self.stdscr.addstr(max_y - 1, 0, " " * (max_x - 1))  # Clear the listening message
            self.stdscr.addstr(max_y - 2, 0, "Input: " + self.input_buffer[: max_x - 5])
        elif key in (curses.KEY_ENTER, 10, 13):
            if self.input_buffer.strip():
                logger.debug(f"Input: {self.input_buffer}")
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
            self.scroll_offset = min(
                self.scroll_offset + 1,
                len(self.interaction_log_manager.interaction_log) if self.active_screen == 1 else len(self.debug_log),
            )
        elif key == curses.KEY_DOWN:
            self.scroll_offset = max(self.scroll_offset - 1, 0)
        elif key == ord("\t"):
            self.stdscr.addstr(max_y - 1, 0, "Enter Private Notes: ")
            curses.echo()
            private_notes = ""
            while True:
                note_key = self.stdscr.getch()
                if note_key in (curses.KEY_ENTER, 10, 13):
                    break
                elif note_key == curses.KEY_BACKSPACE or note_key == 127:
                    private_notes = private_notes[:-1]
                    self.stdscr.addstr(
                        max_y - 1,
                        len("Enter Private Notes: "),
                        " " * (max_x - len("Enter Private Notes: ")),
                    )
                    self.stdscr.addstr(
                        max_y - 1, len("Enter Private Notes: "), private_notes
                    )
                elif 32 <= note_key <= 126:
                    private_notes += chr(note_key)
                    self.stdscr.addstr(
                        max_y - 1, len("Enter Private Notes: "), private_notes
                    )
                self.stdscr.refresh()
            curses.noecho()
            logger.debug(f"Private Notes: {private_notes}")
            # Process private notes (to be handled)
            # ...
        elif 32 <= key <= 126:
            self.input_buffer += chr(key)

        while not self.debug_queue.empty():
            try:
                debug_message = self.debug_queue.get_nowait()
                self.debug_log.append(debug_message)
            except QueueEmpty:
                break

class InteractionProcessor:
    def __init__(self, interaction_queue, state, llama_manager, interaction_log_manager, index_manager):
        self.interaction_queue = interaction_queue
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.index_manager = index_manager

    async def start(self):
        logger.debug("Starting interaction processing...")
        while True:
            try:
                interaction = self.interaction_queue.get_nowait()
                user_input = interaction.get("input", "")
                user_private_notes = interaction.get("private_notes", "")
                audio_waveform = interaction.get("audio_waveform", None)

                logger.info(f"Processing interaction: {user_input}")

                text_features = extract_text_features(user_input)
                if audio_waveform is not None:
                    audio_features = extract_audio_features(audio_waveform)
                    # Combine text and audio features here if needed

                private_notes = await self.llama_manager.generate_private_notes(user_input)
                # Process private notes (to be implemented)
                # ...

                response = await self.llama_manager.generate_llama_response(user_input, private_notes)

                logger.info(f"Response: {response}")
                await self.interaction_log_manager.append(f"Thought: {response}")
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input": user_input,
                    "private_notes": user_private_notes,
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
    def __init__(self, state, llama_manager, interaction_log_manager, event_scheduler):
        self.state = state
        self.llama_manager = llama_manager
        self.interaction_log_manager = interaction_log_manager
        self.event_scheduler = event_scheduler

    async def start(self):
        logger.debug("Starting autonomous thought generation...")
        while True:
            current_hour = datetime.now().hour
            if daily_sleep_start <= current_hour or current_hour < daily_sleep_end:
                logger.debug("System in sleep mode, reducing activity.")
                self.state["is_sleeping"] = True
                await asyncio.sleep(random.uniform(5, 10))
            else:
                self.state["is_sleeping"] = False
                logger.debug("Generating new thoughts")
                # Spawn multiple thought tasks
                thought_tasks = []
                for _ in range(3):  # Number of simultaneous thoughts
                    thought_tasks.append(asyncio.create_task(self.generate_thought()))
                await asyncio.gather(*thought_tasks)
                await asyncio.sleep(random.uniform(1, 3))

    async def generate_thought(self):
        self.state["ongoing_thoughts"] += 1
        response = ""
        try:
            thought_prompt = f"Thought at {datetime.now().strftime('%H:%M:%S')}"
            private_notes = await self.llama_manager.generate_private_notes(thought_prompt)
            # Process private notes (to be implemented)
            # ...
            response = await self.llama_manager.generate_llama_response(thought_prompt, private_notes)
            self.state.setdefault("current_thoughts", []).append(response)
            await self.interaction_log_manager.append(f"Thought: {response}")
            # Simulate processing time
            await asyncio.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}")
            await asyncio.sleep(1)
        finally:
            self.state["ongoing_thoughts"] = max(0, self.state["ongoing_thoughts"] - 1)
            # Remove the thought from current_thoughts
            if "current_thoughts" in self.state and response in self.state["current_thoughts"]:
                self.state["current_thoughts"].remove(response)

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

def transcribe_audio(audio_waveform, sample_rate=16000):
    logger.debug("Transcribing audio...")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperModel.from_pretrained("openai/whisper-small")
    model.eval()

    # Prepare the audio input
    input_features = processor(
        audio_waveform, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0])
    logger.debug(f"Transcription result: {transcription}")
    return transcription

def extract_text_features(text):
    # Placeholder for text feature extraction
    logger.debug("Extracting text features...")
    # Implement your text feature extraction logic here
    return {}

def extract_audio_features(audio_waveform):
    # Placeholder for audio feature extraction
    logger.debug("Extracting audio features...")
    # Implement your audio feature extraction logic here
    return {}

class EventCompressor:
    def __init__(self, llama_manager, event_scheduler):
        self.llama_manager = llama_manager
        self.event_scheduler = event_scheduler

    async def start(self):
        logger.debug("Starting periodic event compression...")
        while True:
            await self.compress_events()
            await asyncio.sleep(3600)  # Run every hour

    async def compress_events(self):
        logger.debug("Starting event compression")
        try:
            with open(HARD_LOG_PATH, "r") as log_file:
                logs = [json.loads(line) for line in log_file]
            events_text = ""
            for entry in logs:
                events_text += f"Task: {entry['input']}\n"
                if entry["private_notes"]:
                    events_text += f"Private Notes: {entry['private_notes']}\n"
                events_text += f"Thought: {entry['output']}\n"
            prompt = f"""Summarize the following interactions, incorporating relevant private notes to improve future training:

{events_text}

Summary:"""
            logger.debug("Generating summary using LLM")
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

class MainApplication:
    def __init__(self):
        self.state = {
            "unprocessed_interactions": 0,
            "ongoing_thoughts": 0,
            "next_event": "Not scheduled",
            "is_sleeping": False,  # New state variable
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
        loop = asyncio.get_running_loop()  # Store the event loop
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

# Run the program
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
