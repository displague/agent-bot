import asyncio
import threading
import json
import time
import os
import requests
import curses
from llama_cpp import Llama

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
        self.restore_state()
        self.restore_timers()
        self.log('System initialized.')

    def log(self, entry):
        with self.lock:
            self.logs.append(entry)
            with open('system_logs.jsonl', 'a') as f:
                if isinstance(entry, str):
                    log_entry = {'type': 'log', 'message': entry, 'timestamp': time.time()}
                else:
                    log_entry = entry
                f.write(json.dumps(log_entry) + '\n')
            # Update the TUI display
            self.update_display()

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
        self.log({
            'type': 'set_timer',
            'delay': delay,
            'callback': callback.__name__,
            'args': args,
            'timestamp': time.time()
        })

    async def fetch_world_data(self):
        # Fetching top news headlines as context
        try:
            response = requests.get(
                'https://www.reddit.com/r/worldnews/top.json?limit=5',
                headers={'User-agent': 'Mozilla/5.0'}
            )
            data = response.json()
            headlines = [item['data']['title'] for item in data['data']['children']]
            self.context = ' '.join(headlines)
            self.log({'type': 'context_update', 'context': self.context, 'timestamp': time.time()})
            self.log('World data fetched and context updated.')
        except Exception as e:
            self.log({'type': 'error', 'message': str(e), 'timestamp': time.time()})

    async def set_annual_goal(self):
        prompt = ("I have a mind and capabilities. I have time to be. "
                  "What will I do first? What's going on in the world? "
                  "What's important to me and how can I drive positive outcomes towards that end.")
        response = self.llm(prompt)
        goal = response['choices'][0]['text'].strip()
        self.long_term_goals.append(goal)
        self.log({'type': 'annual_goal', 'goal': goal, 'timestamp': time.time()})
        self.log(f"Annual goal set: {goal}")

    async def evaluate_progress(self):
        prompt = (f"Given my goals {self.long_term_goals}, and the current context {self.context}, "
                  "how am I progressing towards my goals? Should I adjust my plans?")
        response = self.llm(prompt)
        evaluation = response['choices'][0]['text'].strip()
        self.log({'type': 'evaluation', 'evaluation': evaluation, 'timestamp': time.time()})
        self.log(f"Evaluation: {evaluation}")
        if 'adjust' in evaluation.lower():
            await self.adjust_goals(evaluation)

    async def adjust_goals(self, evaluation):
        prompt = (f"Based on the evaluation: '{evaluation}', what adjustments should I make "
                  "to my goals or plans to better achieve positive outcomes?")
        response = self.llm(prompt)
        adjustments = response['choices'][0]['text'].strip()
        self.long_term_goals.append(adjustments)
        self.log({'type': 'goal_adjustment', 'adjustments': adjustments, 'timestamp': time.time()})
        self.log(f"Goals adjusted: {adjustments}")

    async def interact_with_user(self, user_input):
        self.log({'type': 'user_interaction', 'input': user_input, 'timestamp': time.time()})
        if user_input:
            self.context += ' ' + user_input
            self.log(f"User input received: {user_input}")

    async def monitor_world_events(self):
        while True:
            await self.fetch_world_data()
            # Re-evaluate progress based on new context
            await self.evaluate_progress()
            await asyncio.sleep(3600)  # Wait for an hour before next check

    def update_display(self):
        # Update the TUI display
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()
        log_display_height = max_y - 2  # Leave space for the input prompt
        # Display logs
        start_line = max(0, len(self.logs) - log_display_height)
        for idx, entry in enumerate(self.logs[start_line:]):
            if isinstance(entry, dict):
                message = entry.get('message') or entry.get('evaluation') or entry.get('goal') \
                          or entry.get('adjustments') or entry.get('input') or str(entry)
            else:
                message = str(entry)
            self.stdscr.addnstr(idx, 0, message, max_x)
        # Draw the input prompt
        self.stdscr.addstr(max_y - 1, 0, 'Input: ')
        self.stdscr.refresh()

def main(stdscr):
    # Initialize the Llama model (update the model path as needed)
    llm = Llama(model_path='model.bin')  # Replace with your model's path

    # Initialize the AutonomousSystem
    system = AutonomousSystem(llm, stdscr)

    # Start the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the system's initial tasks
    loop.create_task(system.set_annual_goal())
    loop.create_task(system.monitor_world_events())

    # Handle user input asynchronously
    async def handle_user_input():
        curses.echo()
        while True:
            max_y, max_x = stdscr.getmaxyx()
            stdscr.move(max_y - 1, 7)
            stdscr.clrtoeol()
            user_input = stdscr.getstr().decode('utf-8')
            await system.interact_with_user(user_input)
            system.update_display()

    loop.create_task(handle_user_input())

    # Run the event loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == '__main__':
    curses.wrapper(main)
