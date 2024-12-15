# llama_model_manager.py

import asyncio
import contextlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging

from config import MODEL_PATH, MAX_WORKERS
from llama_cpp import Llama

logger = logging.getLogger("autonomous_system.llama_model_manager")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class LlamaModelManager:
    """
    Manages the Llama model, including model calls and context management.
    Simulates function calling as per Llama-3 specifications.
    """

    def __init__(self, model_path=MODEL_PATH):
        """Initializes the Llama model manager."""
        self.logger = logging.getLogger("autonomous_system.llama_model_manager")
        self.llm = Llama(model_path=model_path)
        self.llm_lock = Lock()
        self.llm_context = []
        self.context_limit = 512
        self.original_stderr = sys.stderr
        self.available_functions = {
            "search_index": self.fn_search_index,
            "schedule_event": self.fn_schedule_event,
        }

    @contextlib.contextmanager
    def capture_llm_stderr(self):
        """Captures stderr output from the Llama model."""
        self.logger.debug("Capturing LLM stderr")
        stderr_fd = self.original_stderr.fileno()
        with open("logs/llm_stderr.log", "w") as f:
            old_stderr = os.dup(stderr_fd)
            os.dup2(f.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                sys.stderr = self.original_stderr

    def llm_call(self, prompt, max_tokens=512):
        """Calls the Llama model with the given prompt."""
        with self.llm_lock, self.capture_llm_stderr():
            self.logger.debug(f"LLM call with prompt: {prompt}")
            response = self.llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
            return response["choices"][0]["text"]

    def estimate_token_count(self, text):
        """Estimates the number of tokens in the given text."""
        tokens = self.llm.tokenize(text.encode("utf-8"))
        return len(tokens)

    def update_context(self, new_entry):
        """Updates the model context with a new entry."""
        self.llm_context.append(new_entry)
        context_str = "\n".join(self.llm_context)
        total_tokens = self.estimate_token_count(context_str)
        while total_tokens > self.context_limit:
            self.llm_context.pop(0)
            context_str = "\n".join(self.llm_context)
            total_tokens = self.estimate_token_count(context_str)

    async def generate_private_notes(self, prompt):
        """Generates private notes for the given prompt."""
        loop = asyncio.get_running_loop()
        analysis_prompt = f"""
Note any uncertainties or hesitations about the following prompt. Keep it concise.

Context:
{prompt}

Notes:"""
        try:
            private_notes = await loop.run_in_executor(
                executor, self.llm_call, analysis_prompt, 150
            )
            self.logger.debug(f"Generated private notes: {private_notes}")
            return private_notes.strip()
        except Exception as e:
            self.logger.error(f"Error generating private notes: {e}")
            return ""

    def call_function(self, function_name, arguments):
        """Calls the appropriate function based on the given function name and arguments."""
        if function_name in self.available_functions:
            self.logger.debug(
                f"Calling function: {function_name} with args: {arguments}"
            )
            return self.available_functions[function_name](**arguments)
        else:
            self.logger.error(f"Function not found: {function_name}")
            return "Error: Function not found."

    def fn_search_index(self, keyword):
        """Simulates searching the index."""
        # Placeholder - actual implementation might need a reference to index_manager
        self.logger.debug(f"Simulating search index for keyword: {keyword}")
        return f"Searched for {keyword}, results: ..."

    def fn_schedule_event(self, event_type, message):
        """Simulates scheduling an event."""
        # Placeholder - in reality would schedule with event_scheduler
        self.logger.debug(
            f"Simulating scheduling event: {event_type} with message: {message}"
        )
        return f"Scheduled {event_type} event with message: {message}"

    async def run_phase(self, phase_name, prompt, notes):
        """Runs a single phase by calling the LLM and handling function calls."""
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
        self.logger.debug(f"Running phase: {phase_name} with prompt: {prompt}")
        response = await loop.run_in_executor(
            executor, self.llm_call, internal_prompt, 512
        )
        response = response.strip()

        if response.startswith("{") and response.endswith("}"):
            try:
                call_data = json.loads(response)
                fname = call_data["name"]
                args = call_data["arguments"]
                function_result = self.call_function(fname, args)
                self.update_context(
                    f"Function Call: {fname}, args: {args}, result: {function_result}"
                )
                return function_result
            except Exception as e:
                self.logger.error(f"Failed to parse function call: {e}")
                return "Error parsing function call."
        else:
            self.update_context(f"{phase_name} Output: {response}")
            return response
