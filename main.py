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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json

# Set up logging for debug
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/application.log"), logging.StreamHandler()],
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

# Redirect stderr to capture llama_cpp output
stderr_fileno = sys.stderr.fileno()
stderr_backup = os.dup(stderr_fileno)
stderr_pipe = os.pipe()
os.dup2(stderr_pipe[1], stderr_fileno)


async def capture_stderr():
    with os.fdopen(stderr_pipe[0]) as stderr_read:
        while True:
            line = stderr_read.readline()
            if line:
                logger.debug(f"[STDERR] Captured stderr: {line.strip()}")
                debug_queue.put(f"[STDERR] {line.strip()}")
            await asyncio.sleep(0.1)


# Constants
daily_sleep_start = 23  # 11 PM
daily_sleep_end = 7  # 7 AM

# Initialize Llama model
logger.debug("Initializing Llama model...")
llm = Llama(model_path="model.bin")  # Update with the correct model path
llm_lock = Lock()

# Shared context for Llama
llm_context = ""

# Queues and state
interaction_queue = Queue()
debug_queue = Queue()
event_queue = asyncio.Queue()
state = {
    "unprocessed_interactions": 0,
    "ongoing_thoughts": 0,
    "next_event": "Not scheduled",
}


# Indexing system
def extract_keywords(text):
    # Simple keyword extraction (could use NLP techniques)
    return list(set(text.lower().split()))


def index_interaction(entry):
    logger.debug("Indexing interaction")
    index = load_index()
    keywords = extract_keywords(entry["user_input"] + " " + entry["assistant_output"])
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
        assistant_message = event["message"]
        logger.debug(f"Assistant (Reminder): {assistant_message}")
        interaction_log.append(f"\nAssistant (Reminder): {assistant_message}\n")
    elif event_type == "lookup":
        keyword = event["keyword"]
        results = search_context(keyword)
        assistant_private_notes = f"Lookup results for '{keyword}': {results}"
        logger.debug(
            f"Assistant's Private Notes after lookup: {assistant_private_notes}"
        )
        # Process results or update context as needed
    elif event_type == "deferred_topic":
        topic = event["topic"]
        logger.debug(f"Assistant revisiting deferred topic: {topic}")
        assistant_message = (
            f"I've thought more about {topic} and would like to discuss it further."
        )
        interaction_log.append(f"\nAssistant: {assistant_message}\n")
    elif event_type == "rag_completed":
        logger.debug("Processing RAG completed event")
        # Schedule training event after a time buffer
        message = "RAG processing has completed. Proceeding with training adjustments."
        training_event = {
            "type": "training",
            "message": message,
            "trigger_time": datetime.now() + timedelta(minutes=5),
        }
        await schedule_event(training_event)
    elif event_type == "training":
        assistant_message = event["message"]
        logger.debug(f"Assistant (Training): {assistant_message}")
        interaction_log.append(f"\nAssistant (Training): {assistant_message}\n")
        # Implement training logic here
    # Update next event in state
    state["next_event"] = "Not scheduled" if event_queue.empty() else "Event pending"


async def schedule_event(event):
    logger.debug(f"Scheduling event: {event}")
    await event_queue.put(event)


# Create an executor for running blocking LLM calls
executor = ThreadPoolExecutor(max_workers=5)

interaction_log_lock = asyncio.Lock()


async def safe_append_interaction_log(entry):
    async with interaction_log_lock:
        interaction_log.append(entry)


# Generate response using Llama model
async def generate_llama_response(prompt, assistant_private_notes):
    global llm_context
    logger.debug(f"Generating response for prompt: {prompt}")
    loop = asyncio.get_event_loop()
    try:
        # Build internal prompt including private notes
        internal_prompt = f"""
# Assistant's Private Notes:
{assistant_private_notes}

# Conversation:
{llm_context}
User: {prompt}
Assistant:"""
        response = await loop.run_in_executor(executor, llm_call, internal_prompt)
        generated_text = response.strip()
        # Update the context with Llama's response
        llm_context += f"\nUser: {prompt}\nAssistant: {generated_text}"
        logger.debug(f"Generated response: {generated_text}")
        return generated_text
    except Exception as e:
        logger.error(f"Error generating response from Llama model: {e}")
        return "Error generating response."


def llm_call(prompt):
    with llm_lock:
        response = llm(prompt, max_tokens=150)
        return response["choices"][0]["text"]


async def generate_assistant_private_notes(prompt):
    logger.debug("Generating assistant's private notes")
    loop = asyncio.get_event_loop()
    try:
        analysis_prompt = f"""
As an AI assistant, briefly note any uncertainties or hesitations you have regarding the following user prompt. Do not provide an answer to the user.

User Prompt:
{prompt}

Assistant's Private Notes (max 50 words):"""
        assistant_private_notes = await loop.run_in_executor(
            executor, llm_call_private_notes, analysis_prompt
        )
        assistant_private_notes = assistant_private_notes.strip()
        logger.debug(f"Assistant's Private Notes: {assistant_private_notes}")
        return assistant_private_notes
    except Exception as e:
        logger.error(f"Error generating assistant's private notes: {e}")
        return ""


def llm_call_private_notes(prompt):
    with llm_lock:
        response = llm(prompt, max_tokens=100, stop=["\n\n"])
        return response["choices"][0]["text"]


def process_private_notes(private_notes, from_assistant=False):
    logger.debug(f"Processing private notes: {private_notes}")
    # Check for triggers in private notes
    if "I should look for previous conversations about" in private_notes:
        keyword = (
            private_notes.split("I should look for previous conversations about")[1]
            .strip()
            .strip(".")
        )
        # Schedule a lookup event
        event = {"type": "lookup", "keyword": keyword, "trigger_time": datetime.now()}
        asyncio.run_coroutine_threadsafe(
            schedule_event(event), asyncio.get_event_loop()
        )
    if "remind me at" in private_notes:
        parts = private_notes.split("remind me at")
        message = parts[0].strip()
        time_str = parts[1].strip().split()[0]  # Simplistic parsing
        reminder_time = parse_time(time_str)
        if reminder_time:
            event = {
                "type": "reminder",
                "message": message,
                "trigger_time": reminder_time,
            }
            asyncio.run_coroutine_threadsafe(
                schedule_event(event), asyncio.get_event_loop()
            )
    if "Let's finish this conversation first" in private_notes and from_assistant:
        start = private_notes.find("I'm specifically interested in")
        end = private_notes.find(". Let's finish this conversation first")
        if start != -1 and end != -1:
            topic = private_notes[
                start + len("I'm specifically interested in") : end
            ].strip()
            # Schedule an event to revisit the topic later
            event = {
                "type": "deferred_topic",
                "topic": topic,
                "trigger_time": datetime.now() + timedelta(days=1),
            }
            asyncio.run_coroutine_threadsafe(
                schedule_event(event), asyncio.get_event_loop()
            )


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
def compress_events():
    logger.debug("Starting event compression")
    try:
        with open(HARD_LOG_PATH, "r") as log_file:
            logs = [json.loads(line) for line in log_file]
        events_text = ""
        for entry in logs:
            events_text += f"User: {entry['user_input']}\n"
            if entry["user_private_notes"]:
                events_text += f"User's Private Notes: {entry['user_private_notes']}\n"
            events_text += f"Assistant: {entry['assistant_output']}\n"
            if entry["assistant_private_notes"]:
                events_text += (
                    f"Assistant's Private Notes: {entry['assistant_private_notes']}\n"
                )
        prompt = f"""Summarize the following interactions, incorporating relevant private notes to improve future training:

{events_text}

Summary:"""
        logger.debug("Generating summary using LLM")
        with llm_lock:
            response = llm(prompt, max_tokens=250)
            summary = response["choices"][0]["text"].strip()
        compressed_entry = {"timestamp": datetime.now().isoformat(), "summary": summary}
        with open(COMPRESSED_LOG_PATH, "a") as comp_log_file:
            comp_log_file.write(json.dumps(compressed_entry) + "\n")
        logger.info("Event compression completed")
        # Trigger an event indicating RAG processing has completed
        rag_event = {"type": "rag_completed", "trigger_time": datetime.now()}
        asyncio.run_coroutine_threadsafe(
            schedule_event(rag_event), asyncio.get_event_loop()
        )
    except Exception as e:
        logger.error(f"Error in compress_events: {e}")


# Curses-based TUI rendering
async def render_tui(stdscr):
    logger.debug("Starting TUI rendering...")
    curses.curs_set(1)
    stdscr.nodelay(True)
    active_screen = 1  # 1 for main screen, 2 for debug screen

    input_buffer = ""
    global interaction_log
    interaction_log = []  # Keep a list of all interactions for scrolling
    debug_log = []  # Keep a list of all debug messages for scrolling
    scroll_offset = 0

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        if active_screen == 1:
            # Draw status bar
            status_bar = f" Unprocessed interactions: {state['unprocessed_interactions']} | Ongoing thoughts: {state['ongoing_thoughts']} | Next event: {state['next_event']} "
            stdscr.addstr(0, 0, status_bar[:max_x], curses.A_REVERSE)

            # Display thoughts and interactions (scrollable view)
            display_log = interaction_log[
                -(max_y - 4 + scroll_offset) : (
                    -scroll_offset if scroll_offset > 0 else None
                )
            ]
            current_y = 1
            for interaction in display_log:
                stdscr.addstr(current_y, 0, interaction[:max_x])
                current_y += 1
        elif active_screen == 2:
            # Display debug output (scrollable view)
            display_log = debug_log[
                -(max_y - 4 + scroll_offset) : (
                    -scroll_offset if scroll_offset > 0 else None
                )
            ]
            current_y = 1
            for debug_message in display_log:
                stdscr.addstr(current_y, 0, debug_message[:max_x])
                current_y += 1

        # Draw input prompt
        stdscr.addstr(max_y - 2, 0, "You: " + input_buffer[: max_x - 5])
        stdscr.refresh()

        # Handle key input to switch screens and handle user input
        key = stdscr.getch()
        if key == 27:  # ESC key to switch screens
            logger.debug("Switching screen view")
            active_screen = 2 if active_screen == 1 else 1
        elif key == curses.KEY_BACKSPACE or key == 127:
            input_buffer = input_buffer[:-1]
        elif key in (curses.KEY_ENTER, 10, 13):
            if input_buffer.strip():
                logger.debug(f"User input: {input_buffer}")
                interaction_queue.put(
                    {"user_input": input_buffer, "user_private_notes": ""}
                )
                state["unprocessed_interactions"] += 1
                interaction_log.append(f"You: {input_buffer}")
                input_buffer = ""
        elif key == curses.KEY_UP:
            scroll_offset = min(
                scroll_offset + 1,
                len(interaction_log) if active_screen == 1 else len(debug_log),
            )
        elif key == curses.KEY_DOWN:
            scroll_offset = max(scroll_offset - 1, 0)
        elif key == ord("\t"):  # Tab key to input private notes
            stdscr.addstr(max_y - 1, 0, "Enter Private Notes: ")
            curses.echo()
            private_notes = stdscr.getstr(
                max_y - 1, len("Enter Private Notes: ")
            ).decode()
            curses.noecho()
            logger.debug(f"User's Private Notes: {private_notes}")
            # Process private notes
            process_private_notes(private_notes)
            interaction_queue.put(
                {"user_input": input_buffer, "user_private_notes": private_notes}
            )
        elif 32 <= key <= 126:  # Printable characters
            input_buffer += chr(key)

        # Add interactions to the log
        await asyncio.sleep(0.1)


# Asynchronous processing of user interactions


# Update process_interactions function to be asynchronous
async def process_interactions():
    logger.debug("Starting interaction processing...")
    while True:
        try:
            interaction = interaction_queue.get_nowait()
            user_input = interaction.get("user_input", "")
            user_private_notes = interaction.get("user_private_notes", "")
            logger.info(f"Processing interaction: {user_input}")
            # Generate assistant's private notes
            assistant_private_notes = await generate_assistant_private_notes(user_input)
            # Process user and assistant private notes
            process_private_notes(user_private_notes)
            process_private_notes(assistant_private_notes, from_assistant=True)
            # Generate assistant's response
            response = await generate_llama_response(
                user_input, assistant_private_notes
            )
            logger.info(f"Response: {response}")
            await safe_append_interaction_log(f"Assistant: {response}")
            # Log interaction
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "user_private_notes": user_private_notes,
                "assistant_output": response,
                "assistant_private_notes": assistant_private_notes,
            }
            with open(HARD_LOG_PATH, "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            index_interaction(log_entry)
            state["unprocessed_interactions"] = max(
                0, state["unprocessed_interactions"] - 1
            )
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Simulate processing delay
        except QueueEmpty:
            await asyncio.sleep(1)  # Polling interval


# Autonomous thought generation
async def chain_of_thought():
    logger.debug("Starting autonomous thought generation...")
    while True:
        current_hour = datetime.now().hour
        if daily_sleep_start <= current_hour or current_hour < daily_sleep_end:
            logger.debug("System in sleep mode, reducing activity.")
            await asyncio.sleep(random.uniform(5, 10))  # Less active during sleep hours
        else:
            logger.debug("Generating a new thought")
            thought_prompt = f"Thought at {datetime.now().strftime('%H:%M:%S')}"
            assistant_private_notes = generate_assistant_private_notes(thought_prompt)
            process_private_notes(assistant_private_notes, from_assistant=True)
            response = generate_llama_response(thought_prompt, assistant_private_notes)
            state["ongoing_thoughts"] += 1
            interaction_log.append(f"Assistant Thought: {response}")
            debug_queue.put(f"Generated thought: {response}")
            await asyncio.sleep(random.uniform(1, 3))


# Main entry point
async def main(stdscr):
    logger.debug("Starting main event loop...")
    tasks = [
        asyncio.create_task(render_tui(stdscr)),
        asyncio.create_task(process_interactions()),
        asyncio.create_task(chain_of_thought()),
        asyncio.create_task(capture_stderr()),
        asyncio.create_task(event_scheduler()),
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Exception in main: {e}")


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
        # Ensure all tasks are canceled
        pending = asyncio.all_tasks()
        for task in pending:
            task.cancel()
        # Wait for tasks to be canceled
        try:
            asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        except Exception as e:
            logger.error(f"Error while cancelling tasks: {e}")
        os.dup2(stderr_backup, stderr_fileno)
        sys.exit(0)
