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

# Initialize Llama model
logger.debug("Initializing Llama model...")
llm = Llama(model_path="model.bin")  # Update with the correct model path
llm_lock = Lock()

# Shared context for Llama
llm_context = ""

interaction_log = []

# Helper function for thread-safe interaction_log updates
interaction_log_lock = asyncio.Lock()

# Queues and state
interaction_queue = Queue()
debug_queue = Queue()
event_queue = asyncio.Queue()
state = {
    "unprocessed_interactions": 0,
    "ongoing_thoughts": 0,
    "next_event": "Not scheduled",
    "is_sleeping": False,  # New state variable
}

# Create an executor for running blocking LLM calls
executor = ThreadPoolExecutor(max_workers=5)


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


def record_audio(duration=5, sample_rate=16000):
    logger.debug("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    waveform = torch.from_numpy(audio_data.flatten())
    logger.debug("Audio recording completed.")
    return waveform


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


# Define context manager for capturing stderr during LLM calls
@contextlib.contextmanager
def capture_llm_stderr():
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


# Indexing system
def extract_keywords(text):
    # Simple keyword extraction (could use NLP techniques)
    return list(set(text.lower().split()))


def index_interaction(entry):
    logger.debug("Indexing interaction")
    index = load_index()
    keywords = extract_keywords(entry["input"] + " " + entry["output"])
    for keyword in keywords:
        if keyword in index:
            index[keyword].append(entry)
        else:
            index[keyword] = [entry]
    save_index(index)


def load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r") as index_file:
            return json.load(index_file)
    return {}


def save_index(index):
    with open(INDEX_PATH, "w") as index_file:
        json.dump(index, index_file)


def search_context(keyword):
    index = load_index()
    return index.get(keyword.lower(), [])


# Event handling
async def event_scheduler():
    logger.debug("Starting event scheduler")
    while True:
        try:
            event = await event_queue.get()
            asyncio.create_task(handle_event(event))
        except Exception as e:
            logger.error(f"Error in event scheduler: {e}")
        await asyncio.sleep(1)


async def handle_event(event):
    event_type = event["type"]
    logger.info(f"Handling event: {event}")
    if event_type == "reminder":
        message = event["message"]
        logger.debug(f"Reminder: {message}")
        await safe_append_interaction_log(f"\nReminder: {message}\n")
    elif event_type == "lookup":
        keyword = event["keyword"]
        results = search_context(keyword)
        notes = f"Lookup results for '{keyword}': {results}"
        logger.debug(f"Notes after lookup: {notes}")
    elif event_type == "deferred_topic":
        topic = event["topic"]
        logger.debug(f"Revisiting deferred topic: {topic}")
        message = (
            f"I've thought more about {topic} and would like to discuss it further."
        )
        await safe_append_interaction_log(f"\nThought: {message}\n")
    elif event_type == "rag_completed":
        logger.debug("Processing RAG completed event")
        message = "RAG processing has completed. Proceeding with training adjustments."
        training_event = {
            "type": "training",
            "message": message,
            "trigger_time": datetime.now() + timedelta(minutes=5),
        }
        await schedule_event(training_event)
    elif event_type == "training":
        message = event["message"]
        logger.debug(f"Training: {message}")
        await safe_append_interaction_log(f"\nTraining: {message}\n")
    state["next_event"] = "Not scheduled" if event_queue.empty() else "Event pending"


async def schedule_event(event):
    logger.debug(f"Scheduling event: {event}")
    await event_queue.put(event)


async def safe_append_interaction_log(entry):
    global interaction_log
    async with interaction_log_lock:
        interaction_log.append(entry)


# Generate response using Llama model
async def generate_llama_response(prompt, notes):
    global llm_context
    logger.debug(f"Generating response for prompt: {prompt}")
    loop = asyncio.get_running_loop()
    try:
        internal_prompt = f"""
# Reflection:
{notes}

# Conversation:
{llm_context}
Task: {prompt}
Thought:"""
        response = await loop.run_in_executor(executor, llm_call, internal_prompt)
        generated_text = response.strip()
        # Update context, ensuring clear separation between inputs and thoughts
        llm_context += f"\nTask: {prompt}\nThought: {generated_text}"
        logger.debug(f"Generated response: {generated_text}")
        return generated_text
    except Exception as e:
        logger.error(f"Error generating response from Llama model: {e}")
        await asyncio.sleep(1)
        return "Error generating response."


def llm_call(prompt):
    with llm_lock, capture_llm_stderr():
        response = llm(prompt, max_tokens=512, stop=["\n\n"])
        return response["choices"][0]["text"]


def llm_call_private_notes(prompt):
    with llm_lock, capture_llm_stderr():
        response = llm(prompt, max_tokens=150, stop=["\n\n"])
        return response["choices"][0]["text"]


async def generate_private_notes(prompt):
    logger.debug("Generating private notes")
    loop = asyncio.get_running_loop()
    try:
        analysis_prompt = f"""
Note any uncertainties or hesitations regarding the following prompt. Keep it concise and relevant.

Context Information:
{prompt}

Notes (max 30 words):"""
        private_notes = await loop.run_in_executor(
            executor, llm_call_private_notes, analysis_prompt
        )
        private_notes = private_notes.strip()
        logger.debug(f"Private Notes: {private_notes}")
        return private_notes
    except Exception as e:
        logger.error(f"Error generating private notes: {e}")
        await asyncio.sleep(1)
        return ""


async def process_private_notes(notes, from_agent=False):
    logger.debug(f"Processing private notes: {notes}")
    if "I should look for previous conversations about" in notes:
        keyword = (
            notes.split("I should look for previous conversations about")[1]
            .strip()
            .strip(".")
        )
        event = {"type": "lookup", "keyword": keyword, "trigger_time": datetime.now()}
        await schedule_event(event)
    if "remind me at" in notes:
        parts = notes.split("remind me at")
        message = parts[0].strip()
        time_str = parts[1].strip().split()[0]
        reminder_time = parse_time(time_str)
        if reminder_time:
            event = {
                "type": "reminder",
                "message": message,
                "trigger_time": reminder_time,
            }
            await schedule_event(event)
    if "Let's finish this conversation first" in notes and from_agent:
        start = notes.find("I'm specifically interested in")
        end = notes.find(". Let's finish this conversation first")
        if start != -1 and end != -1:
            topic = notes[start + len("I'm specifically interested in") : end].strip()
            event = {
                "type": "deferred_topic",
                "topic": topic,
                "trigger_time": datetime.now() + timedelta(days=1),
            }
            await schedule_event(event)


def parse_time(time_str):
    try:
        now = datetime.now()
        reminder_time = datetime.strptime(time_str, "%H:%M")
        reminder_time = reminder_time.replace(
            year=now.year, month=now.month, day=now.day
        )
        if reminder_time < now:
            reminder_time += timedelta(days=1)
        return reminder_time
    except ValueError:
        return None


# Compress events function
async def compress_events():
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
            executor, llm_call, prompt
        )
        summary = summary.strip()
        compressed_entry = {"timestamp": datetime.now().isoformat(), "summary": summary}
        with open(COMPRESSED_LOG_PATH, "a") as comp_log_file:
            comp_log_file.write(json.dumps(compressed_entry) + "\n")
        logger.info("Event compression completed")
        rag_event = {"type": "rag_completed", "trigger_time": datetime.now()}
        await schedule_event(rag_event)
    except Exception as e:
        logger.error(f"Error in compress_events: {e}")
        await asyncio.sleep(1)


# Curses-based TUI rendering
async def render_tui(stdscr):
    logger.debug("Starting TUI rendering...")
    curses.curs_set(1)
    stdscr.nodelay(True)
    active_screen = 1

    input_buffer = ""
    global interaction_log
    debug_log = []
    scroll_offset = 0
    audio_waveform = None  # Initialize here

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        if active_screen == 1:
            sleep_status = "SLEEPING" if state["is_sleeping"] else "ACTIVE"
            status_bar = f" Status: {sleep_status} | Unprocessed: {state['unprocessed_interactions']} | Thoughts: {state['ongoing_thoughts']} | Next event: {state['next_event']} "
            stdscr.addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)

            current_y = 1
            # Display current thoughts
            current_thoughts = state.get("current_thoughts", [])
            for thought in current_thoughts:
                stdscr.addstr(current_y, 0, f"Thought: {thought[:max_x]}")
                current_y += 1

            # Display interaction log
            async with interaction_log_lock:
                display_log = interaction_log[
                    -(max_y - current_y - 2 + scroll_offset) : (
                        -scroll_offset if scroll_offset > 0 else None
                    )
                ]
            for interaction in display_log:
                if current_y >= max_y - 2:
                    break
                stdscr.addstr(current_y, 0, interaction[:max_x])
                current_y += 1
        elif active_screen == 2:
            # Safely access interaction_log
            async with interaction_log_lock:
                display_log = interaction_log[
                    -(max_y - 4 + scroll_offset) : (
                        -scroll_offset if scroll_offset > 0 else None
                    )
                ]
            current_y = 1
            for debug_message in display_log:
                stdscr.addstr(current_y, 0, debug_message[:max_x])
                current_y += 1

        stdscr.addstr(max_y - 2, 0, "Input: " + input_buffer[: max_x - 5])
        stdscr.refresh()

        key = stdscr.getch()
        if key == 27:
            logger.debug("Switching screen view")
            active_screen = 2 if active_screen == 1 else 1
        elif key == curses.KEY_BACKSPACE or key == 127:
            input_buffer = input_buffer[:-1]
        elif key == ord("v"):  # 'v' key for voice input
            stdscr.addstr(max_y - 1, 0, "Listening... (Press any key to stop)")
            stdscr.refresh()
            audio_recorder = AudioRecorder(duration=5)
            audio_recorder.start()
            stdscr.getch()  # Wait for any key press
            sd.stop()  # Stop recording when any key is pressed
            audio_recorder.join()
            audio_waveform = audio_recorder.audio_waveform
            if audio_waveform is not None:
                transcribed_text = transcribe_audio(audio_waveform)
                if transcribed_text.strip():
                    input_buffer = transcribed_text
                else:
                    input_buffer = ""
            else:
                input_buffer = ""
            stdscr.addstr(
                max_y - 1, 0, " " * (max_x - 1)
            )  # Clear the listening message
            stdscr.addstr(max_y - 2, 0, "Input: " + input_buffer[: max_x - 5])
        elif key in (curses.KEY_ENTER, 10, 13):
            if input_buffer.strip():
                logger.debug(f"Input: {input_buffer}")
                interaction_queue.put(
                    {
                        "input": input_buffer,
                        "private_notes": "",
                        "audio_waveform": audio_waveform,
                    }
                )
                state["unprocessed_interactions"] += 1
                await safe_append_interaction_log(f"Input: {input_buffer}")
                input_buffer = ""
        elif key == curses.KEY_UP:
            scroll_offset = min(
                scroll_offset + 1,
                len(interaction_log) if active_screen == 1 else len(debug_log),
            )
        elif key == curses.KEY_DOWN:
            scroll_offset = max(scroll_offset - 1, 0)
        elif key == ord("\t"):
            stdscr.addstr(max_y - 1, 0, "Enter Private Notes: ")
            curses.echo()
            private_notes = ""
            while True:
                note_key = stdscr.getch()
                if note_key in (curses.KEY_ENTER, 10, 13):
                    break
                elif note_key == curses.KEY_BACKSPACE or note_key == 127:
                    private_notes = private_notes[:-1]
                    stdscr.addstr(
                        max_y - 1,
                        len("Enter Private Notes: "),
                        " " * (max_x - len("Enter Private Notes: ")),
                    )
                    stdscr.addstr(
                        max_y - 1, len("Enter Private Notes: "), private_notes
                    )
                elif 32 <= note_key <= 126:
                    private_notes += chr(note_key)
                    stdscr.addstr(
                        max_y - 1, len("Enter Private Notes: "), private_notes
                    )
                stdscr.refresh()
            curses.noecho()
            logger.debug(f"Private Notes: {private_notes}")
            await process_private_notes(private_notes)
            interaction_queue.put(
                {
                    "input": input_buffer,
                    "private_notes": private_notes,
                    "audio_waveform": audio_waveform,
                }
            )
        elif 32 <= key <= 126:
            input_buffer += chr(key)

        while not debug_queue.empty():
            try:
                debug_message = debug_queue.get_nowait()
                debug_log.append(debug_message)
            except QueueEmpty:
                break

        await asyncio.sleep(0.1)


# Asynchronous processing of inputs
async def process_interactions():
    logger.debug("Starting interaction processing...")
    while True:
        try:
            interaction = interaction_queue.get_nowait()
            user_input = interaction.get("input", "")
            user_private_notes = interaction.get("private_notes", "")
            audio_waveform = interaction.get("audio_waveform", None)

            logger.info(f"Processing interaction: {user_input}")

            text_features = extract_text_features(user_input)
            if audio_waveform is not None:
                audio_features = extract_audio_features(audio_waveform)
                # Combine text and audio features here if needed

            private_notes = await generate_private_notes(user_input)
            await process_private_notes(user_private_notes)
            await process_private_notes(private_notes, from_agent=True)

            response = await generate_llama_response(user_input, private_notes)

            logger.info(f"Response: {response}")
            await safe_append_interaction_log(f"Thought: {response}")
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "private_notes": user_private_notes,
                "output": response,
            }
            with open(HARD_LOG_PATH, "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            index_interaction(log_entry)
            state["unprocessed_interactions"] = max(
                0, state["unprocessed_interactions"] - 1
            )
            await asyncio.sleep(random.uniform(0.5, 1.5))
        except QueueEmpty:
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in process_interactions: {e}")
            await asyncio.sleep(1)


# Autonomous thought generation
async def chain_of_thought():
    logger.debug("Starting autonomous thought generation...")
    while True:
        current_hour = datetime.now().hour
        if daily_sleep_start <= current_hour or current_hour < daily_sleep_end:
            logger.debug("System in sleep mode, reducing activity.")
            state["is_sleeping"] = True
            await asyncio.sleep(random.uniform(5, 10))
        else:
            state["is_sleeping"] = False
            logger.debug("Generating new thoughts")
            # Spawn multiple thought tasks
            thought_tasks = []
            for _ in range(3):  # Number of simultaneous thoughts
                thought_tasks.append(asyncio.create_task(generate_thought()))
            await asyncio.gather(*thought_tasks)
            await asyncio.sleep(random.uniform(1, 3))


async def generate_thought():
    state["ongoing_thoughts"] += 1
    try:
        thought_prompt = f"Thought at {datetime.now().strftime('%H:%M:%S')}"
        private_notes = await generate_private_notes(thought_prompt)
        await process_private_notes(private_notes, from_agent=True)
        response = await generate_llama_response(thought_prompt, private_notes)
        state.setdefault("current_thoughts", []).append(response)
        await safe_append_interaction_log(f"Thought: {response}")
        debug_queue.put(f"Generated thought: {response}")
        # Simulate processing time
        await asyncio.sleep(random.uniform(1, 3))
    except Exception as e:
        logger.error(f"Error in generate_thought: {e}")
        await asyncio.sleep(1)
    finally:
        state["ongoing_thoughts"] = max(0, state["ongoing_thoughts"] - 1)
        # Remove the thought from current_thoughts
        if "current_thoughts" in state and response in state["current_thoughts"]:
            state["current_thoughts"].remove(response)


async def periodic_event_compression():
    logger.debug("Starting periodic event compression...")
    while True:
        await compress_events()
        await asyncio.sleep(3600)  # Run every hour


# Main entry point
async def main(stdscr):
    loop = asyncio.get_running_loop()  # Store the event loop
    logger.debug("Starting main event loop...")
    tasks = [
        asyncio.create_task(render_tui(stdscr)),
        asyncio.create_task(process_interactions()),
        asyncio.create_task(chain_of_thought()),
        asyncio.create_task(event_scheduler()),
        asyncio.create_task(periodic_event_compression()),
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
        curses.wrapper(lambda stdscr: asyncio.run(main(stdscr)))
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
