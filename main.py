import os
import json
import datetime
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize the LLaMA model
llm = Llama(model_path=os.environ.get("MODEL_PATH", "model.bin"))

# Ensure directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('compressed_logs', exist_ok=True)
os.makedirs('index', exist_ok=True)

# Paths to files
HARD_LOG_PATH = 'logs/hard_log.jsonl'
COMPRESSED_LOG_PATH = 'compressed_logs/compressed_log.jsonl'
SHARED_CONTEXT_PATH = 'logs/shared_context.json'
EVENT_QUEUE_PATH = 'logs/event_queue.jsonl'
INDEX_PATH = 'index/context_index.json'

# Thread pool executor for running blocking tasks
executor = ThreadPoolExecutor(max_workers=5)

# Function to log interactions
def log_interaction(entry):
    logger.debug(f"Logging interaction: {entry}")
    with open(HARD_LOG_PATH, 'a') as log_file:
        log_file.write(json.dumps(entry) + '\n')
    # Index the interaction for search
    asyncio.run_coroutine_threadsafe(index_interaction_async(entry), asyncio.get_event_loop())

# Indexing system
async def index_interaction_async(entry):
    logger.debug("Indexing interaction asynchronously")
    index = await load_index_async()
    # Simple indexing by keywords (could be enhanced)
    keywords = extract_keywords(entry['user_input'] + ' ' + entry['assistant_output'])
    for keyword in keywords:
        if keyword in index:
            index[keyword].append(entry)
        else:
            index[keyword] = [entry]
    await save_index_async(index)

def extract_keywords(text):
    # Simple keyword extraction (could use NLP techniques)
    return list(set(text.lower().split()))

async def load_index_async():
    if os.path.exists(INDEX_PATH):
        loop = asyncio.get_event_loop()
        with open(INDEX_PATH, 'r') as index_file:
            data = await loop.run_in_executor(executor, index_file.read)
            return json.loads(data)
    return {}

async def save_index_async(index):
    loop = asyncio.get_event_loop()
    data = json.dumps(index)
    with open(INDEX_PATH, 'w') as index_file:
        await loop.run_in_executor(executor, index_file.write, data)

async def search_context_async(keyword):
    index = await load_index_async()
    return index.get(keyword.lower(), [])

# Event handling
async def event_scheduler():
    logger.debug("Starting event scheduler")
    while True:
        await asyncio.sleep(1)  # Check every second
        await process_events_async()

async def process_events_async():
    now = datetime.datetime.now()
    pending_events = await load_events_async()
    remaining_events = []
    for event in pending_events:
        event_time = datetime.datetime.fromisoformat(event['trigger_time'])
        if now >= event_time:
            asyncio.create_task(handle_event_async(event))
        else:
            remaining_events.append(event)
    await save_events_async(remaining_events)

async def handle_event_async(event):
    event_type = event['type']
    logger.info(f"Handling event: {event}")
    if event_type == 'reminder':
        assistant_message = event['message']
        logger.debug(f"Assistant (Reminder): {assistant_message}")
        print(f"\nAssistant (Reminder): {assistant_message}\n")
    elif event_type == 'lookup':
        keyword = event['keyword']
        results = await search_context_async(keyword)
        assistant_private_notes = f"Lookup results for '{keyword}': {results}"
        logger.debug(f"Assistant's Private Notes after lookup: {assistant_private_notes}")
        # You can process results further or update assistant's context
    elif event_type == 'rag_completed':
        logger.debug("Processing RAG completed event")
        # Handle actions that should occur after RAG processing
        # For example, schedule a future event for training
        message = "RAG processing has completed. Proceeding with training adjustments."
        # Schedule training event after a time buffer
        training_event = {
            "type": "training",
            "message": message,
            "trigger_time": (datetime.datetime.now() + datetime.timedelta(minutes=5)).isoformat()
        }
        schedule_event(training_event)
    elif event_type == 'training':
        assistant_message = event['message']
        logger.debug(f"Assistant (Training): {assistant_message}")
        # Proceed with training adjustments based on the RAG output
        # This is where you would integrate training logic
        print(f"\nAssistant (Training): {assistant_message}\n")
    elif event_type == 'deferred_topic':
        # Assistant revisits a topic after deferral
        topic = event['topic']
        logger.debug(f"Assistant revisiting deferred topic: {topic}")
        assistant_message = f"I've thought more about {topic} and would like to discuss it further."
        print(f"\nAssistant: {assistant_message}\n")
        # You can initiate a new conversation or provide additional information
    # Add other event types as needed

async def load_events_async():
    if os.path.exists(EVENT_QUEUE_PATH):
        loop = asyncio.get_event_loop()
        with open(EVENT_QUEUE_PATH, 'r') as event_file:
            data = await loop.run_in_executor(executor, event_file.readlines)
            return [json.loads(line) for line in data]
    return []

async def save_events_async(events):
    loop = asyncio.get_event_loop()
    data = ''.join(json.dumps(event) + '\n' for event in events)
    with open(EVENT_QUEUE_PATH, 'w') as event_file:
        await loop.run_in_executor(executor, event_file.write, data)

def schedule_event(event):
    logger.debug(f"Scheduling event: {event}")
    asyncio.run_coroutine_threadsafe(add_event_async(event), asyncio.get_event_loop())

async def add_event_async(event):
    events = await load_events_async()
    events.append(event)
    await save_events_async(events)

# Function to compress events
async def compress_events_async():
    logger.debug("Starting event compression")
    loop = asyncio.get_event_loop()
    with open(HARD_LOG_PATH, 'r') as log_file:
        data = await loop.run_in_executor(executor, log_file.readlines)
    logs = [json.loads(line) for line in data]
    # Prepare text for summarization, including relevant private notes
    events_text = ""
    for entry in logs:
        events_text += f"User: {entry['user_input']}\n"
        if entry['user_private_notes']:
            events_text += f"User's Private Notes: {entry['user_private_notes']}\n"
        events_text += f"Assistant: {entry['assistant_output']}\n"
        if entry['assistant_private_notes']:
            events_text += f"Assistant's Private Notes: {entry['assistant_private_notes']}\n"
    # Create a prompt for summarization
    prompt = f"""Summarize the following interactions, incorporating relevant private notes to improve future training:

{events_text}

Summary:"""
    logger.debug("Generating summary using LLM")
    summary = await run_in_executor_async(generate_summary, prompt)
    # Save the compressed summary
    compressed_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": summary
    }
    with open(COMPRESSED_LOG_PATH, 'a') as comp_log_file:
        await loop.run_in_executor(executor, comp_log_file.write, json.dumps(compressed_entry) + '\n')
    logger.info("Event compression completed")
    # Trigger an event indicating RAG processing has completed
    rag_event = {
        "type": "rag_completed",
        "trigger_time": datetime.datetime.now().isoformat()
    }
    schedule_event(rag_event)

def generate_summary(prompt):
    response = llm(prompt, max_tokens=2500)
    return response['choices'][0]['text'].strip()

# Helper function to run blocking tasks asynchronously
async def run_in_executor_async(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

# Function to manage contexts
async def get_shared_context_async():
    if os.path.exists(SHARED_CONTEXT_PATH):
        loop = asyncio.get_event_loop()
        with open(SHARED_CONTEXT_PATH, 'r') as context_file:
            data = await loop.run_in_executor(executor, context_file.read)
            return json.loads(data)
    return ""

async def update_shared_context_async(new_entry):
    shared_context = await get_shared_context_async()
    shared_context += new_entry
    data = json.dumps(shared_context)
    with open(SHARED_CONTEXT_PATH, 'w') as context_file:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, context_file.write, data)

async def generate_response_async(user_input, user_private_notes):
    logger.debug("Generating assistant response")
    shared_context = await get_shared_context_async()
    # Build the prompt with contexts
    prompt = f"""
{shared_context}
User: {user_input}
Assistant:"""
    # Agent generates private notes (e.g., hesitations)
    assistant_private_notes = await generate_assistant_private_notes_async(prompt)
    # Check for triggers in user private notes
    await process_private_notes_async(user_private_notes)
    # Build internal prompt including private notes
    internal_prompt = f"""
# Assistant's Private Notes:
{assistant_private_notes}

# Conversation:
{prompt}
"""
    assistant_output = await run_in_executor_async(generate_assistant_output, internal_prompt)
    # Update shared context
    await update_shared_context_async(f"\nUser: {user_input}\nAssistant: {assistant_output}")
    # Prepare log entry
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_input": user_input,
        "user_private_notes": user_private_notes,
        "assistant_output": assistant_output,
        "assistant_private_notes": assistant_private_notes
    }
    # Log the interaction
    log_interaction(log_entry)
    return assistant_output

def generate_assistant_output(prompt):
    response = llm(prompt, max_tokens=150)
    return response['choices'][0]['text'].strip()

async def generate_assistant_private_notes_async(prompt):
    logger.debug("Generating assistant's private notes")
    analysis_prompt = f"""
As the assistant, analyze the following prompt and note any uncertainties or hesitations you have. Do not provide a response to the user yet.

Prompt:
{prompt}

Assistant's Private Notes:"""
    assistant_private_notes = await run_in_executor_async(generate_private_notes, analysis_prompt)
    # Check for triggers in assistant private notes
    await process_private_notes_async(assistant_private_notes, from_assistant=True)
    return assistant_private_notes

def generate_private_notes(prompt):
    response = llm(prompt, max_tokens=100)
    return response['choices'][0]['text'].strip()

async def process_private_notes_async(private_notes, from_assistant=False):
    logger.debug(f"Processing private notes: {private_notes}")
    # Check for triggers in private notes
    if "I should look for previous conversations about" in private_notes:
        keyword = private_notes.split("I should look for previous conversations about")[1].strip().strip('.')
        # Schedule a lookup event
        event = {
            "type": "lookup",
            "keyword": keyword,
            "trigger_time": datetime.datetime.now().isoformat()
        }
        schedule_event(event)
    # Check for reminders or other triggers
    if "remind me at" in private_notes:
        parts = private_notes.split("remind me at")
        message = parts[0].strip()
        time_str = parts[1].strip().split()[0]  # Simple parsing; enhance as needed
        reminder_time = parse_time(time_str)
        if reminder_time:
            event = {
                "type": "reminder",
                "message": message,
                "trigger_time": reminder_time.isoformat()
            }
            schedule_event(event)
    # Handle deferred topics
    if "Let's finish this conversation first" in private_notes and from_assistant:
        # Extract topic of interest
        start = private_notes.find("I'm specifically interested in")
        end = private_notes.find(". Let's finish this conversation first")
        if start != -1 and end != -1:
            topic = private_notes[start + len("I'm specifically interested in"):end].strip()
            # Schedule an event to revisit the topic later
            event = {
                "type": "deferred_topic",
                "topic": topic,
                "trigger_time": (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()
            }
            schedule_event(event)
    # Implement time buffers to prevent infinite loops
    # For example, if an event schedules another event of the same type, ensure there's a delay

def parse_time(time_str):
    try:
        # Parse time in HH:MM format
        now = datetime.datetime.now()
        reminder_time = datetime.datetime.strptime(time_str, '%H:%M')
        reminder_time = reminder_time.replace(year=now.year, month=now.month, day=now.day)
        if reminder_time < now:
            reminder_time += datetime.timedelta(days=1)  # Schedule for next day if time has passed
        return reminder_time
    except ValueError:
        return None

# Main interaction loop
async def main_async():
    # Start event scheduler as a background task
    asyncio.create_task(event_scheduler())
    logger.info("Assistant is ready to interact with the user.")
    print("Welcome to the Event-Based Agentic System. Type 'exit' to quit.\n")
    while True:
        user_input = await run_in_executor_async(input, "User: ")
        if user_input.lower() == 'exit':
            logger.info("User initiated exit.")
            print("Exiting the system.")
            break
        user_private_notes = await run_in_executor_async(input, "User's Private Notes (optional): ")
        assistant_output = await generate_response_async(user_input, user_private_notes)
        print(f"Assistant: {assistant_output}\n")
        # Optionally compress events
        compress_now = await run_in_executor_async(input, "Compress events now? (y/n): ")
        if compress_now.lower() == 'y':
            await compress_events_async()
            print("Events have been compressed.\n")

def main():
    # Run the main_async function in the event loop
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")

if __name__ == "__main__":
    main()
