import asyncio
import threading
import time
import random
import logging
from datetime import datetime
from queue import Queue, Empty as QueueEmpty
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
import requests
from llama_cpp import Llama
import curses

# Set up logging for debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("autonomous_system")

# Constants
daily_sleep_start = 23  # 11 PM
daily_sleep_end = 7  # 7 AM

# Initialize Llama model
llm = Llama(model_path='model.bin')  # Update with the correct model path

# Queues and state
interaction_queue = Queue()
console = Console()
state = {
    "unprocessed_interactions": 0,
    "ongoing_thoughts": 0,
    "next_event": "Not scheduled"
}

# External world data fetcher
def fetch_world_data():
    try:
        response = requests.get('https://www.reddit.com/r/news/top.json?limit=1', headers={'User-agent': 'TUI Bot 0.1'})
        if response.status_code == 200:
            top_article = response.json()['data']['children'][0]['data']['title']
            logger.debug(f"Top News Headline: {top_article}")
            return top_article
        else:
            logger.debug("Failed to fetch world data.")
            return None
    except Exception as e:
        logger.error(f"Error fetching world data: {e}")
        return None

# Generate response using Llama model
def generate_llama_response(prompt):
    try:
        response = llm(prompt)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Error generating response from Llama model: {e}")
        return "Error generating response."

# Curses-based TUI rendering
async def render_tui(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        
        # Draw status bar
        status_bar = f" Unprocessed interactions: {state['unprocessed_interactions']} | Ongoing thoughts: {state['ongoing_thoughts']} | Next event: {state['next_event']} "
        stdscr.addstr(0, 0, status_bar, curses.A_REVERSE)
        
        # Display thoughts and interactions
        current_y = 1
        while not interaction_queue.empty() and current_y < max_y - 2:
            try:
                interaction = interaction_queue.get_nowait()
                stdscr.addstr(current_y, 0, interaction, curses.color_pair(1))
                state["unprocessed_interactions"] = max(0, state["unprocessed_interactions"] - 1)
                current_y += 1
            except QueueEmpty:
                break
        
        # Draw input prompt
        stdscr.addstr(max_y - 1, 0, "You: ")
        stdscr.refresh()
        await asyncio.sleep(0.1)

# Asynchronous processing of user interactions
async def process_interactions():
    while True:
        try:
            interaction = interaction_queue.get_nowait()
            logger.info(f"Processing interaction: {interaction}")
            response = generate_llama_response(interaction)
            logger.info(f"Response: {response}")
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Simulate processing delay
            state["unprocessed_interactions"] = max(0, state["unprocessed_interactions"] - 1)
        except QueueEmpty:
            await asyncio.sleep(1)  # Polling interval

# Autonomous thought generation
async def chain_of_thought():
    while True:
        current_hour = datetime.now().hour
        if daily_sleep_start <= current_hour or current_hour < daily_sleep_end:
            await asyncio.sleep(random.uniform(5, 10))  # Less active during sleep hours
        else:
            logger.debug("Generating a new thought")
            thought_prompt = f"Thought at {datetime.now().strftime('%H:%M:%S')}"
            response = generate_llama_response(thought_prompt)
            interaction_queue.put(response)
            state["ongoing_thoughts"] += 1
            await asyncio.sleep(random.uniform(1, 3))

# Human interaction simulation
def human_input_simulation():
    while True:
        user_input = input("You: ")
        if user_input.strip():
            interaction_queue.put(user_input)
            state["unprocessed_interactions"] += 1

# Start human interaction thread
threading.Thread(target=human_input_simulation, daemon=True).start()

# Main entry point
async def main(stdscr):
    # Set up tasks for autonomous operation and user interaction processing
    tasks = [
        asyncio.create_task(render_tui(stdscr)),
        asyncio.create_task(process_interactions()),
        asyncio.create_task(chain_of_thought()),
    ]
    await asyncio.gather(*tasks)

# Run the program
if __name__ == "__main__":
    try:
        curses.wrapper(lambda stdscr: asyncio.run(main(stdscr)))
    except KeyboardInterrupt:
        console.print("\n[red]Shutting down gracefully...[/red]")