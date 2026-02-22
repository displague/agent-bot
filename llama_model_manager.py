# llama_model_manager.py

import asyncio
import contextlib
import json
import os
import sys
import math
from threading import Lock
import logging
from typing import Any, Callable, Dict, Optional

from config import (
    MODEL_DEFAULT_ALIAS,
    MODEL_LIST,
    MODEL_PATH,
    MODEL_TRANSFORMERS_MAX_NEW_TOKENS,
    MODEL_TRANSFORMERS_OFFLOAD_DIR,
)

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency at runtime
    hf_hub_download = None

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency at runtime
    Llama = None

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoProcessor = None

logger = logging.getLogger("autonomous_system.llama_model_manager")


def _log_prompts_enabled() -> bool:
    value = os.getenv("AGENTBOT_LOG_PROMPTS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


class LlamaModelManager:
    """
    Manages the Llama model, including model calls and context management.
    Simulates function calling as per Llama-3 specifications.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        llm_executor=None,
        status_callback: Optional[Callable[[str], None]] = None,
        voice_loop=None,
    ):
        """Initializes the Llama model manager."""
        self.logger = logging.getLogger("autonomous_system.llama_model_manager")
        self.llm_lock = Lock()
        self.llm_executor = llm_executor
        self.llm_context = []
        self.context_limit = 512
        self.model_path_fallback = model_path
        self.model_catalog = MODEL_LIST
        self.active_model_alias = os.getenv("AGENTBOT_MODEL_ALIAS", MODEL_DEFAULT_ALIAS)
        self.backend = None
        self.llm = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_processor = None
        self.voice_loop = voice_loop
        self.original_stderr = sys.stderr
        self.available_functions = {
            "search_index": self.fn_search_index,
            "schedule_event": self.fn_schedule_event,
            "inspect_audio_snippet": self.fn_inspect_audio_snippet,
        }
        self._status_callback = status_callback
        self._status("Initializing model manager")
        try:
            self.load_model(self.active_model_alias)
        except Exception:
            if self.active_model_alias != MODEL_DEFAULT_ALIAS:
                self.logger.warning(
                    "Failed to load AGENTBOT_MODEL_ALIAS=%s; falling back to %s",
                    self.active_model_alias,
                    MODEL_DEFAULT_ALIAS,
                )
                self._status(
                    f"Model alias '{self.active_model_alias}' failed; falling back to '{MODEL_DEFAULT_ALIAS}'"
                )
                self.active_model_alias = MODEL_DEFAULT_ALIAS
                self.load_model(self.active_model_alias)
            else:
                raise

    def _status(self, message: str) -> None:
        self.logger.info(message)
        if self._status_callback is None:
            return
        try:
            self._status_callback(message)
        except Exception:
            # Never let UI status failures break model loading.
            pass

    def _resolve_alias(self, alias: Optional[str]) -> str:
        if not alias:
            return self.active_model_alias
        cleaned = alias.strip()
        if not cleaned:
            return self.active_model_alias
        return cleaned

    def _resolve_model_spec(self, alias: str) -> Dict[str, Any]:
        if alias not in self.model_catalog:
            raise ValueError(f"Unknown model alias: {alias}")
        spec = dict(self.model_catalog[alias])
        spec.setdefault("alias", alias)
        return spec

    def _resolve_llama_model_path(self, spec: Dict[str, Any]) -> str:
        if self.model_path_fallback and os.path.exists(self.model_path_fallback):
            self._status(f"Using local llama.cpp model: {self.model_path_fallback}")
            return self.model_path_fallback
        repo_id = spec.get("repo_id")
        filename = spec.get("filename")
        if not repo_id or not filename:
            raise RuntimeError(
                "llama_cpp model spec must define repo_id and filename when local MODEL_PATH is absent."
            )
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub is required to resolve GGUF from HF cache. Install huggingface_hub."
            )
        self.logger.info(
            "Resolving llama.cpp model from HF cache: repo=%s file=%s", repo_id, filename
        )
        self._status(f"Resolving GGUF from HF cache: {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _load_llama_cpp(self, spec: Dict[str, Any]) -> None:
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed. Install llama-cpp-python.")
        model_path = self._resolve_llama_model_path(spec)
        requested_ctx = int(os.getenv("AGENTBOT_LLAMA_N_CTX", "2048"))
        self._status(f"Loading llama.cpp backend (n_ctx={requested_ctx})")
        self.llm = Llama(
            model_path=model_path, n_ctx=requested_ctx, n_gpu_layers=-1, n_threads=os.cpu_count()
        )
        self.hf_model = None
        self.hf_tokenizer = None
        self.backend = "llama_cpp"
        self.context_limit = requested_ctx
        self.logger.info(
            "Active model set: alias=%s backend=%s path=%s n_ctx=%s",
            spec["alias"],
            self.backend,
            model_path,
            requested_ctx,
        )

    def _load_transformers(self, spec: Dict[str, Any]) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed. Install transformers and torch."
            )
        repo_id = spec.get("repo_id")
        if not repo_id:
            raise RuntimeError("transformers model spec must define repo_id.")
        self.logger.info("Loading transformers model from HF cache: repo=%s", repo_id)
        self._status(f"Loading tokenizer: {repo_id}")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        if AutoProcessor is not None:
            self._status(f"Loading processor: {repo_id}")
            self.hf_processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        config = None
        if AutoConfig is not None:
            self._status(f"Loading model config: {repo_id}")
            config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        os.makedirs(MODEL_TRANSFORMERS_OFFLOAD_DIR, exist_ok=True)
        self._status(f"Loading transformer weights: {repo_id} (first run may take a while)")
        import torch
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=torch.bfloat16 if PERSONAPLEX_DEVICE == "cuda" else "auto",
            device_map="auto",
            offload_folder=MODEL_TRANSFORMERS_OFFLOAD_DIR,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.llm = None
        self.backend = "transformers"
        self.logger.info("Active model set: alias=%s backend=%s", spec["alias"], self.backend)

    def load_model(self, alias: Optional[str] = None) -> Dict[str, Any]:
        target_alias = self._resolve_alias(alias)
        with self.llm_lock:
            self._status(f"Selecting model alias: {target_alias}")
            spec = self._resolve_model_spec(target_alias)
            backend = str(spec.get("backend", "")).strip().lower()
            if backend == "llama_cpp":
                self._load_llama_cpp(spec)
            elif backend == "transformers":
                self._load_transformers(spec)
            else:
                raise RuntimeError(
                    f"Unsupported backend '{backend}' for alias '{target_alias}'."
                )
            self.active_model_alias = target_alias
            self._status(
                f"Model ready: alias={target_alias} backend={self.backend}"
            )
            return self.get_model_info()

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(spec) for name, spec in self.model_catalog.items()}

    def is_busy(self) -> bool:
        return self.llm_lock.locked()

    def get_model_info(self) -> Dict[str, Any]:
        spec = dict(self.model_catalog.get(self.active_model_alias, {}))
        spec["alias"] = self.active_model_alias
        spec["backend_active"] = self.backend
        return spec

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

    def llm_call(self, prompt, max_tokens=512, audio=None):
        """Calls the Llama model with the given prompt and optional multi-modal data."""
        with self.llm_lock, self.capture_llm_stderr():
            if _log_prompts_enabled():
                self.logger.debug("LLM call prompt: %s", prompt)
            else:
                self.logger.debug(
                    "LLM call invoked (prompt_chars=%s, max_tokens=%s)",
                    len(prompt),
                    max_tokens,
                )
            if self.backend == "llama_cpp":
                safe_prompt, safe_max_tokens = self._fit_prompt_and_budget(
                    prompt, max_tokens
                )
                response = self.llm(
                    safe_prompt,
                    max_tokens=safe_max_tokens,
                    stop=["\n\n"],
                )
                return response["choices"][0]["text"]
            if self.backend == "transformers":
                if self.hf_model is None or self.hf_tokenizer is None:
                    raise RuntimeError("Transformers model is not loaded.")
                
                # Convert context + current prompt into messages
                raw_messages = []
                for entry in self.llm_context:
                    if entry.startswith("Assistant: "):
                        role = "assistant"
                        content = entry[len("Assistant: "):]
                    elif entry.startswith("User: "):
                        role = "user"
                        content = entry[len("User: "):]
                    else:
                        role = "user"
                        content = entry
                    raw_messages.append({"role": role, "content": content})
                
                # Add the current user prompt
                raw_messages.append({"role": "user", "content": prompt})
                
                # Normalize messages: ensure alternating user/assistant and starts with user
                messages = []
                for m in raw_messages:
                    if not messages:
                        if m["role"] == "user":
                            messages.append(m)
                        continue
                    
                    if m["role"] == messages[-1]["role"]:
                        # Merge consecutive same-role messages
                        messages[-1]["content"] += "\n" + m["content"]
                    else:
                        messages.append(m)

                templated_prompt = self.hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Use processor if multi-modal data is present
                if audio is not None and self.hf_processor is not None:
                    model_inputs = self.hf_processor(
                        text=templated_prompt,
                        audios=audio,
                        sampling_rate=16000, # Common for Gemma audio, adjust if needed
                        return_tensors="pt"
                    )
                else:
                    encoded = self.hf_tokenizer(templated_prompt, return_tensors="pt")
                    model_inputs = dict(encoded)
                
                target_device = getattr(self.hf_model, "device", None)
                if target_device is not None:
                    model_inputs = {k: v.to(target_device) for k, v in encoded.items()}
                
                # Ensure floating point model inputs are in bfloat16 if on CUDA
                if target_device is not None and "cuda" in str(target_device):
                    model_inputs = {k: v.to(dtype=torch.bfloat16) if torch.is_floating_point(v) else v 
                                   for k, v in model_inputs.items()}

                output_ids = self.hf_model.generate(
                    **model_inputs,
                    max_new_tokens=min(max_tokens, MODEL_TRANSFORMERS_MAX_NEW_TOKENS),
                    do_sample=True,
                    temperature=0.4,
                    repetition_penalty=1.2,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                )
                new_tokens = output_ids[0][model_inputs["input_ids"].shape[-1] :]
                return self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
            raise RuntimeError("No active model backend is loaded.")

    def _fit_prompt_and_budget(self, prompt: str, max_tokens: int):
        """Ensure llama.cpp calls stay within the configured context window."""
        if self.backend != "llama_cpp" or self.llm is None:
            return prompt, max_tokens
        text = prompt or ""
        reserve = 32
        target_ctx = max(256, int(self.context_limit))
        # Trim by rough character fraction until tokenized input fits.
        for _ in range(8):
            prompt_tokens = self.estimate_token_count(text)
            room = target_ctx - prompt_tokens - reserve
            if room > 8:
                return text, max(8, min(max_tokens, room))
            # Keep the tail of the prompt where latest instruction/context lives.
            ratio = 0.8
            if prompt_tokens > 0:
                ratio = max(0.2, min(0.85, (target_ctx - reserve) / float(prompt_tokens)))
            keep_chars = max(256, int(math.floor(len(text) * ratio)))
            text = text[-keep_chars:]
        # Final fallback: heavily truncate and force tiny generation budget.
        return text[-512:], min(max_tokens, 16)

    def estimate_token_count(self, text):
        """Estimates the number of tokens in the given text."""
        with self.llm_lock:
            if self.backend == "llama_cpp":
                tokens = self.llm.tokenize(text.encode("utf-8"))
                return len(tokens)
            if self.backend == "transformers":
                if self.hf_tokenizer is None:
                    return len((text or "").split())
                return len(self.hf_tokenizer.encode(text or "", add_special_tokens=False))
            return len((text or "").split())

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
                self.llm_executor, self.llm_call, analysis_prompt, 150
            )
            self.logger.debug(
                "Generated private notes (chars=%s)", len(private_notes or "")
            )
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

    def fn_inspect_audio_snippet(self, seconds: float = 5.0, return_raw: bool = False):
        """Inspects the last 'seconds' of audio for non-verbal context."""
        if self.voice_loop is None:
            return "Error: Voice loop not connected to model manager."
        
        audio = self.voice_loop.get_recent_audio(seconds)
        if audio.size == 0:
            return "No audio captured in the buffer yet."
        
        if return_raw:
            return audio

        from utils import extract_audio_features
        features = extract_audio_features(audio)
        return f"Audio Snippet ({seconds}s) Features: {features}"

    async def run_phase(self, phase_name, prompt, notes):
        """Runs a single phase by calling the LLM and handling function calls."""
        loop = asyncio.get_running_loop()

        # Prepare tool definitions
        tools_str = ""
        if self.available_functions:
            tools_str = "# Available Tools:\n"
            for fname, func in self.available_functions.items():
                desc = func.__doc__ or "No description available."
                tools_str += f"- {fname}: {desc}\n"

        # Prepare internal prompt for the phase
        internal_prompt = f"""
# Phase: {phase_name}
# Reflection:
{notes}
{tools_str}
# Context:
{"\n".join(self.llm_context)}
# Instruction:
{prompt}

Now produce the {phase_name} result. If you need to call a function, output JSON in the format:
{{"name": "function_name", "arguments": {{...}}}}
Otherwise, produce the {phase_name} text directly.
"""
        self.logger.debug(
            "Running phase=%s (prompt_chars=%s, context_items=%s)",
            phase_name,
            len(prompt or ""),
            len(self.llm_context),
        )
        response = await loop.run_in_executor(
            self.llm_executor, self.llm_call, internal_prompt, 512
        )
        response = response.strip()

        if response.startswith("{") and response.endswith("}"):
            try:
                call_data = json.loads(response)
                fname = call_data["name"]
                args = call_data["arguments"]
                function_result = self.call_function(fname, args)
                
                # If tool returns raw audio, re-invoke LLM with multi-modal data
                if isinstance(function_result, np.ndarray):
                    self.logger.info("Tool returned raw audio, re-invoking LLM with multi-modal context.")
                    follow_up_prompt = f"I have retrieved the audio you requested. What can you hear in this {len(function_result)/16000:.1f}s snippet?"
                    return await loop.run_in_executor(
                        self.llm_executor, self.llm_call, follow_up_prompt, 512, function_result
                    )

                # Function calls are still part of internal state context for the next phase
                self.update_context(
                    f"Function Call: {fname}, args: {args}, result: {function_result}"
                )
                return function_result
            except Exception as e:
                self.logger.error(f"Failed to parse function call: {e}")
                return "Error parsing function call."
        else:
            # We no longer add every phase output to self.llm_context here.
            # FunctionalAgent will handle the primary dialogue history.
            return response
