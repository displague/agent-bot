import asyncio
import threading
import json
import time
import os
import requests
import curses
import logging
import io
from datetime import datetime, time as dt_time
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
        self.next_scheduled_event = None
        self.sleeping = False  # Initialize sleeping before calling methods that use it
        self.restore_state()
        self.restore_timers()
        self.log('System initialized.')

    def log(self, entry, debug=False):
        with self.lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(entry, str):
                log_entry = {'type': 'log', 'message': entry, 'timestamp': timestamp}
            else:
                log_entry = entry
                log_entry['timestamp'] = timestamp
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

    def restore_timers(self):
        current_time = time.time()
        for entry in self.logs:
            if entry.get('type') == 'set_timer':
                elapsed = current_time - entry['timestamp']
                remaining = entry['delay'] - elapsed
                if remaining > 0:
                    asyncio.create_task(self.schedule_callback(remaining, entry['callback'], entry.get('args', ())))

    async def schedule_callback(self, delay, callback_name, args=()):
        await asyncio.sleep(delay)
        callback = getattr(self, callback_name)
        await callback(*args)

    def set_timer(self, delay, callback, args=()):
        asyncio.create_task(self.schedule_callback(delay, callback.__name__, args))
        self.next_scheduled_event = datetime.now() + timedelta(seconds=delay)
        self.log({
            'type': 'set_timer',
            'delay': delay,
            'callback': callback.__name__,
            'args': args,
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
            await asyncio.sleep(10)  # Process inputs every 10 seconds
            if self.user_input_queue.empty():
                continue
            batch_inputs = []
            while not self.user_input_queue.empty():
                user_input = self.user_input_queue.get()
                batch_inputs.append(user_input)
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
        if len(inputs) > 0:
            prompt = f"User provided insights: {combined_input}. How should I adjust?"
            response = self.llm(prompt)
            thoughts = response['choices'][0]['text'].strip()
            self.log({'type': 'system_thought', 'thoughts': thoughts})
            self.log(f"System thought: {thoughts}")

    async def monitor_world_events(self):
        while True:
            await self.fetch_world_data()
            # Re-evaluate progress based on new context
            task = asyncio.create_task(self.evaluate_progress())
            self.self_reflection_tasks.add(task)
            task.add_done_callback(self.self_reflection_tasks.discard)
            # Dynamic sleep time based on activity
            sleep_duration = self.dynamic_sleep_time()
            await asyncio.sleep(sleep_duration)

    def dynamic_sleep_time(self):
        # Determine sleep duration based on time of day
        current_hour = datetime.now().hour
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
                message = entry.get('message') or entry.get('evaluation') or entry.get('goal') \
                          or entry.get('adjustments') or entry.get('input') or entry.get('thoughts') or str(entry)
            else:
                message = str(entry)
            self.stdscr.addnstr(idx, 0, message, max_x)
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
        if self.next_scheduled_event:
            time_until_event = self.next_scheduled_event - datetime.now()
            event_info = f" | Next Event In: {time_until_event}"
        else:
            event_info = " | No Upcoming Events"
        sleep_status = "Sleeping" if self.sleeping else "Active"
        status = status_bar + event_info + f" | Status: {sleep_status} "
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addnstr(max_y - 2, 0, status.ljust(max_x), max_x)
        self.stdscr.attroff(curses.A_REVERSE)

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
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # For status bar
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # For debug messages

    # Setup logging for llama_cpp
    log_stream = setup_logging()

    # Initialize the Llama model (update the model path as needed)
    llm = Llama(model_path='model.bin')  # Use your model's path

    # Initialize the AutonomousSystem
    system = AutonomousSystem(llm, stdscr)

    # Start the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the system's initial tasks
    loop.create_task(system.set_annual_goal())
    loop.create_task(system.monitor_world_events())
    loop.create_task(system.process_user_inputs())

    # Handle user input asynchronously
    async def handle_user_input():
        curses.echo()
        while True:
            max_y, max_x = stdscr.getmaxyx()
            stdscr.move(max_y - 1, 7)
            stdscr.clrtoeol()
            user_input = stdscr.getstr().decode('utf-8')
            if user_input:
                system.user_input_queue.put(user_input)
            system.update_display()
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
