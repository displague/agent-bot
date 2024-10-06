import asyncio
import threading
import json
import time
import os
import requests
import curses
import logging
import io
import re
from datetime import datetime, timedelta, time as dt_time
from llama_cpp import Llama
from queue import Queue

class AutonomousSystem:
    def __init__(self, llm, stdscr):
        self.llm = llm
        self.stdscr = stdscr
        self.long_term_goals = []
        self.short_term_plans = []
        self.context = ''
        self.logs = []
        self.lock = threading.Lock()
        self.timers = []
        self.user_input_queue = Queue()
        self.self_reflection_tasks = set()
        self.sleeping = False  # Initialize sleeping before calling methods that use it
        self.upcoming_events = []
        self.restore_state()
        self.log('System initialized.')

    def log(self, entry, debug=False):
        with self.lock:
            timestamp = time.time()
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(entry, str):
                log_entry = {'type': 'log', 'message': entry, 'timestamp': timestamp, 'timestamp_str': timestamp_str}
            else:
                if 'timestamp' not in entry:
                    entry['timestamp'] = timestamp
                if 'timestamp_str' not in entry:
                    entry['timestamp_str'] = timestamp_str
                log_entry = entry
            self.logs.append(log_entry)
            with open('system_logs.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            # Update the TUI display
            self.update_display(debug_message=entry if debug else None)

    def restore_state(self):
        if not os.path.exists('system_logs.jsonl'):
            return
        with open('system_logs.jsonl', 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.logs.append(entry)
                if entry.get('type') == 'annual_goal':
                    self.long_term_goals.append(entry['goal'])
                elif entry.get('type') == 'short_term_plan':
                    self.short_term_plans.append(entry['plan'])
                elif entry.get('type') == 'context_update':
                    self.context = entry['context']

    async def restore_timers(self):
        current_time = time.time()
        for entry in self.logs:
            if entry.get('type') == 'set_timer':
                entry_timestamp = entry['timestamp']
                if isinstance(entry_timestamp, str):
                    # Try to parse the string timestamp
                    try:
                        # First, attempt to parse as a float string
                        entry_timestamp = float(entry_timestamp)
                    except ValueError:
                        # If that fails, try to parse as a datetime string
                        try:
                            dt = datetime.strptime(entry_timestamp, '%Y-%m-%d %H:%M:%S')
                            entry_timestamp = dt.timestamp()
                        except ValueError:
                            # Could not parse timestamp; skip this entry
                            continue
                elapsed = current_time - entry_timestamp
                remaining = entry['delay'] - elapsed
                if remaining > 0:
                    asyncio.create_task(self.schedule_callback(remaining, entry['callback'], entry.get('args', ())))
                else:
                    # Event has already passed
                    pass

    async def schedule_callback(self, delay, callback_name, args=()):
        await asyncio.sleep(delay)
        callback = getattr(self, callback_name)
        await callback(*args)

    def set_timer(self, delay, callback, args=()):
        asyncio.create_task(self.schedule_callback(delay, callback.__name__, args))
        event_time = datetime.now() + timedelta(seconds=delay)
        self.upcoming_events.append(event_time)
        self.log({
            'type': 'set_timer',
            'delay': delay,
            'callback': callback.__name__,
            'args': args,
            'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': time.time()
        })

    async def fetch_world_data(self):
        if self.should_sleep():
            self.log('System is sleeping. Skipping world data fetch.')
            return
        try:
            response = requests.get(
                'https://www.reddit.com/r/worldnews/top.json?limit=5',
                headers={'User-agent': 'Mozilla/5.0'}
            )
            data = response.json()
            headlines = [item['data']['title'] for item in data['data']['children']]
            self.context = ' '.join(headlines)
            self.log({'type': 'context_update', 'context': self.context})
            self.log('World data fetched and context updated.')
        except Exception as e:
            self.log({'type': 'error', 'message': str(e)})

    async def set_annual_goal(self):
        if self.should_sleep():
            self.log('System is sleeping. Skipping annual goal setting.')
            return
        prompt = ("I have a mind and capabilities. I have time to be. "
                  "What will I do first? What's going on in the world? "
                  "What's important to me and how can I drive positive outcomes towards that end.")
        response = self.llm(prompt)
        goal = response['choices'][0]['text'].strip()
        self.long_term_goals.append(goal)
        self.log({'type': 'annual_goal', 'goal': goal})
        self.log(f"Annual goal set: {goal}")

    async def evaluate_progress(self):
        if self.should_sleep():
            self.log('System is sleeping. Skipping progress evaluation.')
            return
        prompt = (f"Given my goals {self.long_term_goals}, and the current context {self.context}, "
                  "how am I progressing towards my goals? Should I adjust my plans?")
        response = self.llm(prompt)
        evaluation = response['choices'][0]['text'].strip()
        self.log({'type': 'evaluation', 'evaluation': evaluation})
        self.log(f"Evaluation: {evaluation}")
        if 'adjust' in evaluation.lower():
            await self.adjust_goals(evaluation)

    async def adjust_goals(self, evaluation):
        prompt = (f"Based on the evaluation: '{evaluation}', what adjustments should I make "
                  "to my goals or plans to better achieve positive outcomes?")
        response = self.llm(prompt)
        adjustments = response['choices'][0]['text'].strip()
        self.long_term_goals.append(adjustments)
        self.log({'type': 'goal_adjustment', 'adjustments': adjustments})
        self.log(f"Goals adjusted: {adjustments}")

    async def process_user_inputs(self):
        while True:
            await asyncio.sleep(5)  # Process inputs every 5 seconds
            if self.user_input_queue.empty():
                continue
            batch_inputs = []
            while not self.user_input_queue.empty():
                user_input = self.user_input_queue.get()
                if user_input.strip():
                    batch_inputs.append(user_input)
            if batch_inputs:
                await self.handle_user_inputs(batch_inputs)

    async def handle_user_inputs(self, inputs):
        if self.should_sleep():
            self.log('System is sleeping. User inputs will be processed later.')
            return
        self.log(f"Processing {len(inputs)} user inputs.")
        # Combine inputs for processing
        combined_input = ' '.join(inputs)
        self.context += ' ' + combined_input
        self.log({'type': 'user_interaction', 'input': combined_input})
        # Optionally, the system may choose to respond
        prompt = f"User provided insights: {combined_input}. How should I adjust?"
        response = self.llm(prompt)
        thoughts = response['choices'][0]['text'].strip()
        self.log({'type': 'system_thought', 'thoughts': thoughts})
        self.log(f"System thought: {thoughts}")
        # Check if a reminder needs to be set
        await self.check_for_reminders(thoughts)

    async def check_for_reminders(self, text):
        # Simple regex to find phrases like "in X minutes"
        match = re.search(r'in (\d+) (minute|minutes|hour|hours)', text, re.IGNORECASE)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            if 'hour' in unit:
                delay = amount * 3600
            else:
                delay = amount * 60
            self.log(f"Setting a reminder in {amount} {unit}.")
            self.set_timer(delay, self.reminder_callback)
        else:
            self.log("No reminders to set.")

    async def reminder_callback(self):
        self.log("Reminder: Time to check the latest news.")
        await self.fetch_world_data()
        await self.evaluate_progress()

    async def monitor_world_events(self):
        while True:
            await self.fetch_world_data()
            await asyncio.sleep(600)  # Check every 10 minutes

    async def chain_of_thought(self):
        while True:
            if self.should_sleep():
                self.log('System is sleeping. Pausing chain of thought.')
                await asyncio.sleep(60)
                continue
            prompt = (f"Current goals: {self.long_term_goals}. "
                      f"Context: {self.context}. "
                      "Generate a thought to advance towards my goals.")
            response = self.llm(prompt)
            thought = response['choices'][0]['text'].strip()
            self.log({'type': 'chain_of_thought', 'thought': thought})
            self.log(f"Chain of Thought: {thought}")
            await asyncio.sleep(30)  # Generate thoughts every 30 seconds

    def dynamic_sleep_time(self):
        # Determine sleep duration based on time of day
        if self.should_sleep():
            return 60 * 60  # Sleep for an hour during sleep time
        else:
            return 60 * 10  # Active during the day, check every 10 minutes

    def should_sleep(self):
        # Define sleep hours (e.g., between 11 PM and 7 AM)
        current_time = datetime.now().time()
        start_sleep = dt_time(23, 0)
        end_sleep = dt_time(7, 0)
        if start_sleep <= current_time or current_time <= end_sleep:
            self.sleeping = True
            return True
        else:
            self.sleeping = False
            return False

    def update_display(self, debug_message=None):
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()
        log_display_height = max_y - 3  # Leave space for the status bar and input prompt
        # Display logs
        start_line = max(0, len(self.logs) - log_display_height)
        for idx, entry in enumerate(self.logs[start_line:]):
            if isinstance(entry, dict):
                message_type = entry.get('type')
                message = entry.get('message') or entry.get('evaluation') or entry.get('goal') \
                          or entry.get('adjustments') or entry.get('input') or entry.get('thoughts') \
                          or entry.get('thought') or str(entry)
            else:
                message_type = 'log'
                message = str(entry)
            # Determine color based on message type
            if message_type == 'user_interaction':
                color = curses.color_pair(3)
            elif message_type in ['system_thought', 'evaluation', 'goal_adjustment', 'annual_goal', 'chain_of_thought']:
                color = curses.color_pair(4)
            elif message_type == 'error':
                color = curses.color_pair(5)
            else:
                color = curses.color_pair(0)  # Default color
            self.stdscr.addnstr(idx, 0, message, max_x, color)
        # Display debug messages
        if debug_message:
            self.stdscr.addnstr(log_display_height - 1, 0, str(debug_message), max_x, curses.color_pair(2))
        # Draw the status bar
        self.draw_status_bar()
        # Draw the input prompt
        self.stdscr.addstr(max_y - 1, 0, 'Input: ')
        self.stdscr.refresh()

    def draw_status_bar(self):
        max_y, max_x = self.stdscr.getmaxyx()
        status_bar = f" Unprocessed Inputs: {self.user_input_queue.qsize()} | Self-Reflections: {len(self.self_reflection_tasks)}"
        if self.upcoming_events:
            next_event_time = min(self.upcoming_events)
            time_until_event = next_event_time - datetime.now()
            seconds = int(time_until_event.total_seconds())
            if seconds > 0:
                event_info = f" | Next Event In: {str(timedelta(seconds=seconds))}"
            else:
                event_info = " | Next Event Soon"
        else:
            event_info = " | No Upcoming Events"
        sleep_status = "Sleeping" if self.sleeping else "Active"
        status = status_bar + event_info + f" | Status: {sleep_status} "
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addnstr(max_y - 2, 0, status.ljust(max_x), max_x)
        self.stdscr.attroff(curses.A_REVERSE)
        self.stdscr.refresh()

    async def refresh_display_periodically(self):
        while True:
            self.update_display()
            await asyncio.sleep(1)  # Refresh display every second

def setup_logging():
    # Redirect llama_cpp debug output to a string buffer
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger('llama_cpp')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return log_stream

def main(stdscr):
    # Initialize curses colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # For status bar
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # For debug messages
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)   # For user inputs
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)  # For system thoughts
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)    # For errors

    # Setup logging for llama_cpp
    log_stream = setup_logging()

    # Initialize the Llama model (update the model path as needed)
    llm = Llama(model_path='model.bin')  # Use your model's path

    # Start the event loop BEFORE creating the AutonomousSystem
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize the AutonomousSystem
    system = AutonomousSystem(llm, stdscr)

    # Schedule restore_timers after the event loop is running
    loop.create_task(system.restore_timers())

    # Run the system's initial tasks
    loop.create_task(system.set_annual_goal())
    loop.create_task(system.monitor_world_events())
    loop.create_task(system.process_user_inputs())
    loop.create_task(system.chain_of_thought())
    loop.create_task(system.refresh_display_periodically())

    # Handle user input asynchronously
    async def handle_user_input():
        curses.echo()
        while True:
            max_y, max_x = stdscr.getmaxyx()
            stdscr.move(max_y - 1, 7)
            stdscr.clrtoeol()
            user_input = stdscr.getstr().decode('utf-8')
            if user_input.strip():
                system.user_input_queue.put(user_input)
            await asyncio.sleep(0)

    # Monitor llama_cpp debug output
    async def monitor_debug_output():
        while True:
            await asyncio.sleep(1)
            debug_output = log_stream.getvalue()
            if debug_output:
                system.log(debug_output.strip(), debug=True)
                log_stream.truncate(0)
                log_stream.seek(0)

    loop.create_task(handle_user_input())
    loop.create_task(monitor_debug_output())

    # Run the event loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == '__main__':
    curses.wrapper(main)
