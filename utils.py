import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import collections
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from mss import mss
from PIL import Image

from config import (
    PERSONAPLEX_SERVER_LOG_PATH,
    PERSONAPLEX_CPU_OFFLOAD,
    PERSONAPLEX_DEVICE,
    PERSONAPLEX_USE_CUDA_GRAPHS,
    PERSONAPLEX_OPTIMIZE,
    PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS,
    PERSONAPLEX_PYTHON_BIN,
    PERSONAPLEX_TEXT_PROMPT,
    PERSONAPLEX_VOICE_PROMPT,
    PERSONAPLEX_VOICE_PROMPT_DIR,
    VOICE_SAMPLE_RATE,
    VOICE_CHUNK_SECONDS,
)
import config as _config

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - dependency may be optional at runtime
    sd = None
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dependency may be optional at runtime
    load_dotenv = None

# Global placeholders for lazily loaded modules
moshi_run_inference = None
MimiModel = None
LMGen = None
loaders = None
sentencepiece = None

logger = logging.getLogger("autonomous_system.utils")


_voice_prompt_cache: dict = {}


def _streaming_state_to_cpu(state):
    """Recursively clone a streaming state dict/dataclass, moving tensors to CPU RAM."""
    import copy
    from dataclasses import fields, is_dataclass
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if is_dataclass(state) and not isinstance(state, type):
        new = copy.copy(state)  # shallow copy of the dataclass shell
        for f in fields(state):
            setattr(new, f.name, _streaming_state_to_cpu(getattr(state, f.name)))
        return new
    if isinstance(state, dict):
        return {k: _streaming_state_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_streaming_state_to_cpu(v) for v in state]
    return copy.copy(state)  # primitives / None


def _streaming_state_to_device(state, device):
    """Recursively clone a CPU streaming state, moving tensors to *device*."""
    import copy
    from dataclasses import fields, is_dataclass
    if isinstance(state, torch.Tensor):
        return state.to(device, non_blocking=True).clone()
    if is_dataclass(state) and not isinstance(state, type):
        new = copy.copy(state)
        for f in fields(state):
            setattr(new, f.name, _streaming_state_to_device(getattr(state, f.name), device))
        return new
    if isinstance(state, dict):
        return {k: _streaming_state_to_device(v, device) for k, v in state.items()}
    if isinstance(state, list):
        return [_streaming_state_to_device(v, device) for v in state]
    return copy.copy(state)


def _ensure_voice_prompt_exists(voice_name: str, repo_id: str = "nvidia/personaplex-7b-v1") -> str:
    """Ensure the voice prompt exists locally, downloading from HF if needed."""
    if voice_name in _voice_prompt_cache:
        return _voice_prompt_cache[voice_name]

    from huggingface_hub import hf_hub_download, list_repo_files
    import tarfile

    # If it's an absolute path and exists, use it
    v_path = Path(voice_name)
    if v_path.is_absolute() and v_path.exists():
        _voice_prompt_cache[voice_name] = str(v_path)
        return str(v_path)

    # Check common local locations — voices/ first to avoid walking the whole tree
    project_root = Path(__file__).parent.absolute()
    v_dir = PERSONAPLEX_VOICE_PROMPT_DIR.strip()
    
    local_search_dirs = [project_root / "voices", project_root / "personaplex", Path(".")]
    if v_dir:
        local_search_dirs.insert(0, Path(v_dir))
        
    for search_dir in local_search_dirs:
        if not search_dir.exists():
            continue
        # Recursive search for the filename
        for root, _, files in os.walk(search_dir):
            if voice_name in files:
                found_path = str(Path(root) / voice_name)
                logger.debug("Found voice prompt locally at: %s", found_path)
                _voice_prompt_cache[voice_name] = found_path
                return found_path

    # Not found locally, attempt HF download
    logger.info("Voice prompt '%s' not found locally. Attempting HF download...", voice_name)
    try:
        files = list_repo_files(repo_id=repo_id)
        if voice_name in files:
            result = hf_hub_download(repo_id=repo_id, filename=voice_name)
            _voice_prompt_cache[voice_name] = result
            return result
        if f"voices/{voice_name}" in files:
            result = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice_name}")
            _voice_prompt_cache[voice_name] = result
            return result
        
        # Fallback to voices.tgz
        if "voices.tgz" in files:
            logger.info("Prompt not found as standalone. Downloading and extracting voices.tgz...")
            tgz_path = hf_hub_download(repo_id=repo_id, filename="voices.tgz")
            project_root = Path(__file__).parent.absolute()
            extract_dir = project_root / "voices"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            
            # Find it in extracted dir
            for root, _, files in os.walk(extract_dir):
                if voice_name in files:
                    result = str(Path(root) / voice_name)
                    _voice_prompt_cache[voice_name] = result
                    return result
    except Exception as e:
        logger.error("Failed to download voice prompt from HF: %s", e)
    
    return voice_name


class PersonaPlexStreamingSession:
    """Manages an active, stateful streaming session with PersonaPlex models."""
    
    def __init__(self, manager, text_prompt: str, voice_prompt_path: str):
        self.manager = manager
        self.text_prompt = text_prompt
        self.voice_prompt_path = voice_prompt_path
        self.lm_gen = None
        self.frame_size = 0
        self._lock = threading.Lock()
        self.all_text_tokens = []
        self._is_ready = False

    def start(self):
        """Initialize the session by restoring the pre-warmed lm_gen state.

        Uses _restore_primed_state() instead of re-running step_system_prompts
        (~46s) so session starts are instant after load() warmup.
        """
        self.manager._apply_optimizations()
        self.manager.load()

        with torch.no_grad(), self.manager._lock:
            self.lm_gen = self.manager.lm_gen
            self.frame_size = int(self.manager.mimi.sample_rate / self.manager.mimi.frame_rate)

            if not self.manager._restore_primed_state():
                # _lm_primed_state not available yet — fall back to full setup
                from moshi.offline import wrap_with_system_tags
                final_voice_prompt = _ensure_voice_prompt_exists(self.voice_prompt_path)
                if final_voice_prompt.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
                else:
                    self.lm_gen.load_voice_prompt(final_voice_prompt)
                self.lm_gen.text_prompt_tokens = self.manager.text_tokenizer.encode(
                    wrap_with_system_tags(self.text_prompt)
                )
                self.manager.mimi.reset_streaming()
                self.manager.other_mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                self.lm_gen.step_system_prompts(self.manager.mimi)
                self.manager.mimi.reset_streaming()
                self.manager._save_primed_state()

            self._is_ready = True

    def step(self, audio_chunk: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Process a single audio chunk and return generated audio and text (if any)."""
        if not self._is_ready:
            self.start()
            
        # Use manager._lock so step() is mutually exclusive with infer_stream/infer.
        with torch.no_grad(), self.manager._lock:
            all_out_pcms = []
            new_texts = []
            
            for i in range(0, len(audio_chunk), self.frame_size):
                sub_chunk = audio_chunk[i : i + self.frame_size]
                if len(sub_chunk) < self.frame_size:
                    sub_chunk = np.pad(sub_chunk, (0, self.frame_size - len(sub_chunk)))
                
                # Use pre-computed silence tokens for near-zero frames — avoids a
                # GPU mimi.encode() call per silent frame (saves ~1ms each).
                if np.abs(sub_chunk).max() < 1e-5:
                    from moshi.models.lm import SILENCE_TOKENS
                    codes = torch.as_tensor(
                        SILENCE_TOKENS, dtype=torch.long, device=self.manager.device
                    ).view(1, 8, 1)
                    _ = self.manager.other_mimi.encode(
                        torch.zeros(1, 1, self.frame_size, device=self.manager.device)
                    )  # keep other_mimi codec state in sync
                else:
                    sub_chunk = np.ascontiguousarray(sub_chunk)
                    chunk_ts = torch.from_numpy(sub_chunk).to(self.manager.device).unsqueeze(0).unsqueeze(0)
                    codes = self.manager.mimi.encode(chunk_ts)
                    _ = self.manager.other_mimi.encode(chunk_ts)  # state sync only — discard
                
                tokens = self.lm_gen.step(codes)
                
                text_token = tokens[0, 0].item()
                if text_token not in {0, 1, 2, 3}:
                    piece = self.manager.text_tokenizer.IdToPiece(text_token)
                    self.all_text_tokens.append(piece)
                    new_texts.append(piece)
                
                # Use mimi (not other_mimi) for decode — mimi holds the correct audio codec state.
                # other_mimi.decode() is called as a discard to keep its state in sync.
                out_pcm = self.manager.mimi.decode(tokens[:, 1 : self.manager.lm.dep_q + 1])
                _ = self.manager.other_mimi.decode(tokens[:, 1 : self.manager.lm.dep_q + 1])
                all_out_pcms.append(out_pcm.cpu().detach().numpy().squeeze())
            
            final_audio = np.concatenate(all_out_pcms) if all_out_pcms else None
            final_text = "".join(new_texts) if new_texts else None
            
            return final_audio, final_text


class PersonaPlexManager:
    """Manages persistent in-process PersonaPlex models for fast inference."""
    
    def __init__(self, device: str = PERSONAPLEX_DEVICE, cpu_offload: bool = PERSONAPLEX_CPU_OFFLOAD, status_callback: Optional[Callable[[str], None]] = None):
        self.device = device
        self.cpu_offload = cpu_offload
        self.status_callback = status_callback
        self.mimi = None
        self.other_mimi = None
        self.lm = None
        self.lm_gen = None  # Warm generator
        self.text_tokenizer = None
        self.repo = "nvidia/personaplex-7b-v1"
        self._lock = threading.Lock()
        # Primed-state tracking: True when step_system_prompts has been run and
        # the lm_gen is ready for inference without re-running setup.
        self._primed = False
        self._primed_for: Optional[tuple] = None  # (voice_prompt_path, text_prompt)
        # Saved LMGen streaming state (KV cache) after step_system_prompts so
        # each call can restore it in milliseconds instead of re-running setup.
        self._lm_primed_state = None
        self._optimizations_applied = False
        # Dedicated executor for sequential, low-jitter inference
        from concurrent.futures import ThreadPoolExecutor
        self.step_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="personaplex_step")

    def shutdown(self):
        """Shut down the manager and its executors."""
        if hasattr(self, "step_executor"):
            self.step_executor.shutdown(wait=True)
            logger.info("PersonaPlexManager: step_executor shut down.")

    def _status(self, message: str):
        logger.info(message)
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def _apply_optimizations(self):
        """Apply torch.compile and CUDA graph optimizations based on config."""
        if self._optimizations_applied:
            return
        self._optimizations_applied = True
        opt = PERSONAPLEX_OPTIMIZE.lower()
        
        # 1. Set environment variables based on strategy
        use_compile = (opt in ["auto", "compile", "graphs"])
        use_graphs = (opt in ["auto", "graphs"])

        if use_compile:
            logger.info("PersonaPlexManager: enabling torch.compile (unsetting NO_TORCH_COMPILE).")
            os.environ.pop("NO_TORCH_COMPILE", None)
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            logger.info("PersonaPlexManager: disabling torch.compile.")
            os.environ["NO_TORCH_COMPILE"] = "1"
            os.environ["TORCH_COMPILE_DISABLE"] = "1"

        if use_graphs:
            logger.info("PersonaPlexManager: allowing CUDA graphs (unsetting NO_CUDA_GRAPH).")
            os.environ.pop("NO_CUDA_GRAPH", None)
            self._patch_cuda_graphs(disable=False)
        else:
            logger.info("PersonaPlexManager: disabling CUDA graphs.")
            os.environ["NO_CUDA_GRAPH"] = "1"
            self._patch_cuda_graphs(disable=True)

        # 2. Fix Moshi's broken streaming propagation logic (even in eager mode)
        try:
            import moshi.modules.streaming
            if not getattr(moshi.modules.streaming.StreamingModule, "_is_patched", False):
                def patched_apply_named_streaming(self, fn):
                    def _handle_module(module_inner: torch.nn.Module, prefix: str = "", recurse: bool = True, is_root: bool = False):
                        propagate = True
                        if isinstance(module_inner, moshi.modules.streaming.StreamingModule):
                            if module_inner._streaming_propagate or is_root:
                                fn(prefix, module_inner)
                            else:
                                propagate = False
                        if not recurse:
                            return
                        if propagate:
                            for name, child in module_inner.named_children():
                                new_prefix = (prefix + "." + name) if prefix else name
                                _handle_module(child, prefix=new_prefix)

                    _handle_module(self, is_root=True, recurse=False)
                    for name, child in self.named_children():
                        _handle_module(child, prefix=name)
                
                moshi.modules.streaming.StreamingModule._apply_named_streaming = patched_apply_named_streaming
                moshi.modules.streaming.StreamingModule._is_patched = True
                logger.info("PersonaPlexManager: patched StreamingModule._apply_named_streaming.")
        except Exception as e:
            logger.warning("PersonaPlexManager: failed to patch moshi StreamingModule: %s", e)

    def _patch_cuda_graphs(self, disable: bool):
        """Patch moshi.models.lm.CUDAGraphed to set the disable flag."""
        self._last_patch_disable = disable
        try:
            import moshi.models.lm
            if getattr(moshi.models.lm.CUDAGraphed, "_is_patched", False):
                return
            original_init = moshi.models.lm.CUDAGraphed.__init__
            def patched_init(self, func, warmup_steps=1, disable_orig=False, **kwargs):
                # We ignore the model's 'disable' hint and use our global one
                target_disable = kwargs.get('disable', disable_orig)
                original_init(self, func, warmup_steps=warmup_steps, disable=disable)
            moshi.models.lm.CUDAGraphed.__init__ = patched_init
            moshi.models.lm.CUDAGraphed._is_patched = True
            logger.info("PersonaPlexManager: patched moshi CUDAGraphed.")
        except Exception as e:
            logger.warning("PersonaPlexManager: failed to patch moshi CUDA graphs: %s", e)

    def _save_primed_state(self):
        """Snapshot the lm_gen streaming state after step_system_prompts.

        Tensors are moved to CPU RAM so the full VRAM stays available for
        inference. Restoring clones them back to GPU on demand.
        Must be called inside self._lock.
        """
        try:
            raw = self.lm_gen.get_streaming_state()
            self._lm_primed_state = _streaming_state_to_cpu(raw)
            logger.debug("PersonaPlexManager: saved primed lm_gen streaming state (CPU).")
        except Exception as e:
            logger.warning("PersonaPlexManager: could not save primed state: %s", e)
            self._lm_primed_state = None

    def _restore_primed_state(self) -> bool:
        """Restore lm_gen to the voice-prompt-primed state instantly.

        Returns True if state was restored (no step_system_prompts needed),
        False if _lm_primed_state is not yet available.
        Must be called inside self._lock.
        """
        if self._lm_primed_state is None:
            return False
        try:
            # Ensure any stale KV cache from a previous turn is wiped before
            # loading the clean primed state.
            self.lm_gen.reset_streaming()
            
            gpu_state = _streaming_state_to_device(self._lm_primed_state, self.device)
            self.lm_gen.set_streaming_state(gpu_state)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            logger.info("PersonaPlexManager: restored pre-warmed lm_gen state (no step_system_prompts needed).")
            return True
        except Exception as e:
            logger.warning("PersonaPlexManager: state restore failed, will re-run step_system_prompts: %s", e)
            self._lm_primed_state = None
            return False

    def load(self):
        """Load models into VRAM if not already loaded."""
        global MimiModel, LMGen, loaders, sentencepiece, moshi_run_inference
        
        if self.mimi is not None:
            return
        
        # Apply optimizations BEFORE importing anything from moshi
        self._apply_optimizations()
        
        try:
            from moshi.offline import run_inference as moshi_run_inference_loaded
            from moshi.models import MimiModel as MimiModel_loaded, LMGen as LMGen_loaded, loaders as loaders_loaded
            import sentencepiece as sentencepiece_loaded
            
            moshi_run_inference = moshi_run_inference_loaded
            MimiModel = MimiModel_loaded
            LMGen = LMGen_loaded
            loaders = loaders_loaded
            sentencepiece = sentencepiece_loaded
        except ImportError:
            logger.error("moshi package not available for in-process loading.")
            raise RuntimeError("moshi package not available for in-process loading.")

        def get_vram():
            if torch is not None and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            return 0.0

        vram_before = get_vram()
        self._status("PersonaPlexManager: starting in-process model load...")
        with self._lock:
            from huggingface_hub import hf_hub_download
            
            try:
                # 1) Load Mimi
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading mimi...")
                mimi_weight = hf_hub_download(self.repo, loaders.MIMI_NAME)
                self.mimi = loaders.get_mimi(mimi_weight, self.device)
                self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: mimi loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                vram_before = vram_now
                
                # 2) Load tokenizer
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading tokenizer...")
                tokenizer_path = hf_hub_download(self.repo, loaders.TEXT_TOKENIZER_NAME)
                self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: tokenizer loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                vram_before = vram_now
                
                # 3) Load Moshi LM
                start = time.perf_counter()
                self._status("PersonaPlexManager: loading moshi lm...")
                moshi_weight = hf_hub_download(self.repo, loaders.MOSHI_NAME)
                # Ensure device is correctly passed and cpu_offload is handled
                self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=self.cpu_offload)
                self.lm.eval()
                # Optional quantization — reduces VRAM from ~14 GB to ~8-10 GB (8bit) or ~5-7 GB (4bit)
                # NOTE: Empirically only ~0.2 GB is freed on PersonaPlex because Moshi's large
                # weight matrices live in fused-ops layers (gating/linear_in/linear_out/out_proj)
                # that must be skipped. Use --quantize only for experimentation until a bitsandbytes-
                # or GGUF-based approach is implemented.
                quantize_type = getattr(_config, "PERSONAPLEX_QUANTIZE", "")
                if quantize_type and quantize_type not in ("none", "None"):
                    from quantize import quantize_model_after_load
                    self._status(f"PersonaPlexManager: applying {quantize_type} quantization (note: VRAM reduction may be minimal for Moshi architecture)…")
                    self.lm = quantize_model_after_load(self.lm, quantize_type, device=self.device)
                elif not self.cpu_offload:
                    self.lm.to(self.device)
                dur = time.perf_counter() - start
                vram_now = get_vram()
                self._status(f"PersonaPlexManager: moshi loaded in {dur:.1f}s (VRAM: {vram_now:.2f}GB, +{vram_now-vram_before:.2f}GB)")
                
                # Streaming forever setup
                self.mimi.streaming_forever(1)
                self.other_mimi.streaming_forever(1)
                
                # Warm generator setup
                self.lm_gen = LMGen(
                    self.lm,
                    audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                    sample_rate=self.mimi.sample_rate,
                    device=self.device,
                    frame_rate=self.mimi.frame_rate,
                    save_voice_prompt_embeddings=False,
                    use_sampling=True,
                    temp=0.8,
                    temp_text=0.7,
                    top_k=250,       # MLX default; top_k=1 caused greedy decoding → always same output
                    top_k_text=25,   # MLX default; top_k_text=1 caused repetitive text tokens
                )
                self.lm_gen.streaming_forever(1)
                
                self._status("PersonaPlexManager: models and warm generator loaded and ready.")
                
                # Pre-warm voice prompt so the first inference skips the 26s setup.
                try:
                    from moshi.offline import wrap_with_system_tags
                    self._status("PersonaPlexManager: warming up voice prompt (first inference will be instant)...")
                    voice_prompt_path = PERSONAPLEX_VOICE_PROMPT
                    final_voice_prompt = _ensure_voice_prompt_exists(voice_prompt_path)
                    with torch.no_grad():
                        if final_voice_prompt.endswith('.pt'):
                            self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
                        else:
                            self.lm_gen.load_voice_prompt(final_voice_prompt)
                        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                            wrap_with_system_tags(PERSONAPLEX_TEXT_PROMPT)
                        )
                        self.mimi.reset_streaming()
                        self.other_mimi.reset_streaming()
                        self.lm_gen.reset_streaming()
                        self.lm_gen.step_system_prompts(self.mimi)
                        self.mimi.reset_streaming()  # Reset mimi encode state (matches offline.py behavior)
                    self._primed = True
                    self._primed_for = voice_prompt_path  # Only voice identity matters for primed check
                    self._save_primed_state()  # Snapshot KV cache for fast restore on every call
                    self._status("PersonaPlexManager: voice prompt warmed up and ready.")
                except Exception as warm_err:
                    logger.warning("PersonaPlexManager: voice prompt warmup failed (will warm on first inference): %s", warm_err)
                    self._primed = False
            except Exception as e:
                logger.exception("PersonaPlexManager: failed to load models: %s", e)
                raise

    def create_session(self, text_prompt: str, voice_prompt_path: str) -> PersonaPlexStreamingSession:
        """Create a new streaming session."""
        return PersonaPlexStreamingSession(self, text_prompt, voice_prompt_path)

    def get_status(self) -> Dict[str, Any]:
        """Return detailed status of the manager and models."""
        with self._lock:
            return {
                "loaded": self.mimi is not None,
                "device": self.device,
                "cpu_offload": self.cpu_offload,
                "optimize_strategy": PERSONAPLEX_OPTIMIZE,
                "torch_compile": os.environ.get("NO_TORCH_COMPILE") != "1",
                "cuda_graphs": getattr(self, "_last_patch_disable", None) is False,
            }

    async def infer_async(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Run a single inference turn asynchronously."""
        return await asyncio.to_thread(self.infer, text_prompt, voice_prompt_path, input_wav_path, output_wav_path, output_text_path)

    def infer_stream(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str):
        """Generator that yields audio chunks as they are produced."""
        import moshi.models.lm
        self._apply_optimizations()
        self.load()
        from moshi.offline import wrap_with_system_tags
        
        final_text_prompt = text_prompt or PERSONAPLEX_TEXT_PROMPT
        final_voice_prompt = _ensure_voice_prompt_exists(voice_prompt_path)

        with torch.no_grad(), self._lock:
            if self._restore_primed_state():
                # KV cache restored; just update utterance tokens
                self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(final_text_prompt))
            else:
                # Full setup: load prompts, reset streaming state, run warmup
                if final_voice_prompt.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
                else:
                    self.lm_gen.load_voice_prompt(final_voice_prompt)
                self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(final_text_prompt))
                self.mimi.reset_streaming()
                self.other_mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                self.lm_gen.step_system_prompts(self.mimi)
                self.mimi.reset_streaming()  # Reset mimi encode state (matches offline.py behavior)
                self._save_primed_state()
            
            # Process input WAV
            input_data, _ = sf.read(input_wav_path, dtype="float32")
            if input_data.ndim > 1:
                input_data = input_data.mean(-1)
            
            frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            chunk_size = frame_size
            for i in range(0, len(input_data), chunk_size):
                chunk = input_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                chunk = np.ascontiguousarray(chunk)
                chunk_ts = torch.from_numpy(chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                codes = self.mimi.encode(chunk_ts)
                _ = self.other_mimi.encode(chunk_ts)  # state sync only — discard
                
                tokens = self.lm_gen.step(codes)
                if tokens is None:
                    continue
                
                # Use mimi for decode; other_mimi discard keeps its state in sync
                out_pcm = self.mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                _ = self.other_mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                pcm_np = out_pcm.cpu().detach().numpy().squeeze()
                yield pcm_np

    def tts_stream(self, text: str, voice_prompt_path: str):
        """Generator that yields audio chunks for the given text using teacher-forced TTS.

        Unlike infer_stream (which feeds silence and waits for the model to speak freely),
        this method forces text tokens token-by-token via lm_gen.step(text_token=tok).
        The depformer then generates audio conditioned on knowing exactly what is being said.
        This is the same mechanism used internally by step_system_prompts/_step_text_prompt
        but with the audio output captured for playback.
        """
        self._apply_optimizations()
        self.load()

        final_voice_prompt = _ensure_voice_prompt_exists(voice_prompt_path)

        with torch.no_grad(), self._lock:
            if self._restore_primed_state():
                pass  # KV cache restored; mimi/other_mimi already reset inside helper
            else:
                # First call or restore failed: run full setup then save state for next time
                if final_voice_prompt.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
                else:
                    self.lm_gen.load_voice_prompt(final_voice_prompt)
                self.mimi.reset_streaming()
                self.other_mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                self.lm_gen.step_system_prompts(self.mimi)
                self.mimi.reset_streaming()
                self._save_primed_state()

            # Encode the text WITHOUT system tags — these are the words to speak
            text_tokens = self.text_tokenizer.encode(text)
            # Pre-computed sine token tensor for user audio channel
            sine_frame = self.lm_gen._encode_sine_frame()  # user "input" for context [1,8,1]

            # Teacher-force each text token; do NOT provide moshi_tokens so the depformer
            # samples audio autoregressively (providing zero_frame would force silence output).
            for text_token in text_tokens:
                tokens = self.lm_gen.step(
                    text_token=text_token,
                    input_tokens=sine_frame,
                )
                if tokens is None:
                    continue
                out_pcm = self.mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                pcm_np = out_pcm.cpu().detach().numpy().squeeze()
                logger.debug("tts_stream: frame amp max=%.6f tok=%d",
                             float(np.abs(pcm_np).max()), text_token)
                yield pcm_np

            # Run a few silence frames to flush trailing audio (codec has lookahead delay)
            for _ in range(self.lm_gen.audio_silence_frame_cnt):
                tokens = self.lm_gen.step(
                    text_token=self.lm_gen.zero_text_code,
                    input_tokens=sine_frame,
                )
                if tokens is None:
                    continue
                out_pcm = self.mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                yield out_pcm.cpu().detach().numpy().squeeze()

    def infer(self, text_prompt: str, voice_prompt_path: str, input_wav_path: str, output_wav_path: str, output_text_path: Optional[str] = None):
        """Synchronous inference implementation using the warm models."""
        logger.info("PersonaPlexManager: starting inference for prompt: %s", text_prompt[:100])
        import moshi.models.lm
        
        # Apply optimizations based on strategy
        self._apply_optimizations()
        
        self.load()
        from moshi.offline import warmup, wrap_with_system_tags
        
        # Use provided prompt or default
        final_text_prompt = text_prompt or PERSONAPLEX_TEXT_PROMPT
        
        # Resolve voice prompt path
        final_voice_prompt = _ensure_voice_prompt_exists(voice_prompt_path)
            
        logger.debug("PersonaPlexManager: using voice prompt path: %s", final_voice_prompt)

        with torch.no_grad(), self._lock:
            if self._restore_primed_state():
                # KV cache restored; just update utterance tokens
                self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(final_text_prompt))
                if PERSONAPLEX_USE_CUDA_GRAPHS:
                    frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
                    warmup(self.mimi, self.other_mimi, self.lm_gen, self.device, frame_size)
            else:
                # Full setup: load prompts, reset streaming, run warmup
                if PERSONAPLEX_USE_CUDA_GRAPHS:
                    frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
                    warmup(self.mimi, self.other_mimi, self.lm_gen, self.device, frame_size)
                if final_voice_prompt.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(final_voice_prompt)
                else:
                    self.lm_gen.load_voice_prompt(final_voice_prompt)
                self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(final_text_prompt))
                self.mimi.reset_streaming()
                self.other_mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                self.lm_gen.step_system_prompts(self.mimi)
                self.mimi.reset_streaming()  # Reset mimi encode state (matches offline.py behavior)
                self._save_primed_state()
            
            # Process input WAV
            import soundfile as sf
            input_data, _ = sf.read(input_wav_path, dtype="float32")
            if input_data.ndim > 1:
                input_data = input_data.mean(-1)
            
            frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            logger.info("PersonaPlexManager: processing input audio (%d samples)...", len(input_data))
            all_out_pcms = []
            all_text_tokens = []
            chunk_size = frame_size
            for i in range(0, len(input_data), chunk_size):
                chunk = input_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Ensure contiguous for torch.from_numpy
                chunk = np.ascontiguousarray(chunk)
                chunk_ts = torch.from_numpy(chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                codes = self.mimi.encode(chunk_ts)
                _ = self.other_mimi.encode(chunk_ts)  # state sync only — discard
                
                # Single step
                tokens = self.lm_gen.step(codes)
                
                # Capture text tokens (k=0)
                text_token = tokens[0, 0].item()
                if text_token not in {0, 1, 2, 3}: # Skip BOS/EOS/PAD
                    piece = self.text_tokenizer.IdToPiece(text_token)
                    all_text_tokens.append(piece)

                # Use mimi for decode; other_mimi discard keeps its state in sync
                out_pcm = self.mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                _ = self.other_mimi.decode(tokens[:, 1 : self.lm.dep_q + 1])
                all_out_pcms.append(out_pcm.cpu().detach().numpy().squeeze())
            
            if all_out_pcms:
                res = np.concatenate(all_out_pcms)
                sf.write(output_wav_path, res, self.mimi.sample_rate)
                logger.info("PersonaPlexManager: saved generated audio to %s", output_wav_path)
            
            if output_text_path:
                with open(output_text_path, "w", encoding="utf-8") as f:
                    json.dump(all_text_tokens, f)
                logger.debug("PersonaPlexManager: saved generated text tokens to %s", output_text_path)
            
            return output_wav_path

    def hear_stream(self, heard_text: str, voice_prompt_path: str, user_wav_path: str = None):
        """Conversational inference: feed user speech to PersonaPlex, yield agent PCM frames.

        If *user_wav_path* is given, that audio file is used as the user's voice input
        (any format — converted via ffmpeg if needed).  Otherwise *heard_text* is
        synthesised to speech via text_to_wav().

        Unlike tts_stream (which teacher-forces the LM to say specific words),
        this uses the full conversational path: the model *hears* the user
        speaking and generates its natural response in the PersonaPlex voice.

        Yields numpy float32 arrays (one per mimi frame, 1920 samples @ 24 kHz).
        """
        import tempfile as _tempfile

        _TRAILING_SILENCE_S = 4.0   # seconds of silence after speech — gives model time to respond
        _LEADING_SILENCE_AMP = 0.02  # skip leading output frames that are nearly silent
        _sample_rate = int(self.mimi.sample_rate)

        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
            user_wav = _f.name
        with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
            agent_wav = _f.name
        try:
            if user_wav_path:
                _convert_to_wav(user_wav_path, user_wav, sample_rate=_sample_rate)
            else:
                text_to_wav(heard_text, user_wav, sample_rate=_sample_rate)

            # Append trailing silence so the model has time to formulate a response.
            # Without this, infer() only processes ~2s of audio (~25 frames) —
            # not enough for the model to generate a reply.
            tts_data, tts_sr = sf.read(user_wav, dtype="float32")
            user_speech_samples = len(tts_data)  # track where user speech ends
            trailing = np.zeros(int(tts_sr * _TRAILING_SILENCE_S), dtype=np.float32)
            sf.write(user_wav, np.concatenate([tts_data, trailing]), tts_sr)

            self.infer(
                text_prompt=PERSONAPLEX_TEXT_PROMPT,
                voice_prompt_path=voice_prompt_path,
                input_wav_path=user_wav,
                output_wav_path=agent_wav,
            )
            if os.path.exists(agent_wav):
                data, _ = sf.read(agent_wav, dtype="float32")
                frame = int(self.mimi.sample_rate / self.mimi.frame_rate)
                # In full-duplex mode the model generates one output frame per input
                # frame — including while the user is still speaking. Those "overlap"
                # frames are the model trying to talk over the user and are almost
                # always incoherent. Only play output from the silence window
                # (after user speech ends) where the model actually responds.
                speech_frame_start = (user_speech_samples // frame) * frame
                found_speech = False
                for i in range(speech_frame_start, len(data), frame):
                    chunk = data[i : i + frame]
                    if not found_speech and float(np.abs(chunk).max()) < _LEADING_SILENCE_AMP:
                        continue  # skip leading-silence frames in the response window
                    found_speech = True
                    yield chunk.astype(np.float32)
        finally:
            for p in (user_wav, agent_wav):
                if os.path.exists(p):
                    os.unlink(p)


class AudioMultiplexer:
    """Captures microphone audio and broadcasts chunks to multiple subscribers."""
    
    def __init__(self, sample_rate: int = VOICE_SAMPLE_RATE, chunk_seconds: float = VOICE_CHUNK_SECONDS):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.subscribers: Set[Tuple[asyncio.Queue, Optional[asyncio.AbstractEventLoop]]] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        """Create a new queue and add it to subscribers."""
        queue = asyncio.Queue()
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        with self._lock:
            self.subscribers.add((queue, loop))
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a queue from subscribers."""
        with self._lock:
            to_remove = [s for s in self.subscribers if s[0] == queue]
            for s in to_remove:
                self.subscribers.remove(s)

    def start(self):
        """Start the background capture thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_capture, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_capture(self):
        """Dedicated thread for sounddevice capture to ensure steady timing."""
        logger.info("AudioMultiplexer: starting capture thread.")
        if sd is None:
            logger.error("sounddevice not installed. Audio capture disabled.")
            return

        def callback(indata, frames, time, status):
            if status:
                logger.warning("AudioMultiplexer status: %s", status)
            
            chunk = indata.copy().squeeze()
            with self._lock:
                for queue, loop in self.subscribers:
                    if loop:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    else:
                        try:
                            queue.put_nowait(chunk)
                        except Exception:
                            pass

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, blocksize=int(self.sample_rate * self.chunk_seconds)):
                while not self._stop_event.is_set():
                    self._stop_event.wait(0.1)
        except Exception as e:
            logger.error("AudioMultiplexer capture error: %s", e)
        finally:
            logger.info("AudioMultiplexer: capture thread stopped.")


class RollingAudioBuffer:
    """Subscriber to AudioMultiplexer that maintains a rolling window of recent audio."""
    
    def __init__(self, multiplexer: AudioMultiplexer, max_seconds: float = 10.0):
        self.multiplexer = multiplexer
        self.max_seconds = max_seconds
        self.sample_rate = multiplexer.sample_rate
        self.buffer = collections.deque(maxlen=int(max_seconds / multiplexer.chunk_seconds))
        self._queue = None
        self._task = None
        self._stop_event = asyncio.Event()

    async def start(self):
        self._queue = self.multiplexer.subscribe()
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        logger.info("RollingAudioBuffer: started.")

    async def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._queue:
            self.multiplexer.unsubscribe(self._queue)
        logger.info("RollingAudioBuffer: stopped.")

    async def _run(self):
        while not self._stop_event.is_set():
            try:
                chunk = await self._queue.get()
                self.buffer.append(chunk)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("RollingAudioBuffer error: %s", e)

    def get_audio(self) -> np.ndarray:
        """Return the current rolling buffer as a single numpy array."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))


def set_audio_devices(input_id: Optional[int] = None, output_id: Optional[int] = None):
    """Set the default sounddevice input and output device IDs."""
    if sd is None:
        return
    if input_id is not None:
        sd.default.device[0] = input_id
    if output_id is not None:
        sd.default.device[1] = output_id
    logger.info("Audio devices updated: input=%s, output=%s", sd.default.device[0], sd.default.device[1])


def play_test_tone(duration: float = 1.0, freq: float = 440.0):
    """Play a test tone to verify audio output hardware."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot play tone.")
    
    # Log current default or set device
    dev_idx = sd.default.device[1]
    logger.info("Playing test tone on output device index: %s", dev_idx)
    
    samples = int(duration * VOICE_SAMPLE_RATE)
    t = np.linspace(0, duration, samples, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    sd.play(tone.astype(np.float32), samplerate=VOICE_SAMPLE_RATE)
    sd.wait()


class DebugServer:
    """A lightweight JSON socket server for live interaction and monitoring."""
    def __init__(self, host="127.0.0.1", port=9999, command_handler=None, state=None):
        self.host = host
        self.port = port
        self.command_handler = command_handler
        self.state = state
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        logger.info(f"DebugServer: listening on {self.host}:{self.port}")
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        logger.debug(f"DebugServer: accepted connection from {addr}")
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                line = data.decode().strip()
                if not line:
                    continue
                
                try:
                    payload = json.loads(line)
                    msg_type = payload.get("type", "command")
                    
                    if msg_type == "command" and self.command_handler:
                        cmd = payload.get("data")
                        logger.info(f"DebugServer: injecting command: {cmd}")
                        # Wrap in task to not block the reader
                        asyncio.create_task(self.command_handler(cmd))
                        writer.write(json.dumps({"status": "ok", "message": f"Command '{cmd}' enqueued"}).encode() + b"\n")
                    elif msg_type == "state":
                        # Convert state to something JSON-serializable if needed
                        serializable_state = {k: v for k, v in self.state.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                        writer.write(json.dumps({"status": "ok", "state": serializable_state}).encode() + b"\n")
                    else:
                        writer.write(json.dumps({"status": "error", "message": "Unknown request type"}).encode() + b"\n")
                except Exception as e:
                    writer.write(json.dumps({"status": "error", "message": str(e)}).encode() + b"\n")
                
                await writer.drain()
        except Exception as e:
            logger.error(f"DebugServer client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


def _convert_to_wav(src_path: str, dst_path: str, sample_rate: int = 24000) -> None:
    """Convert *src_path* (any ffmpeg-supported format) to mono WAV at *sample_rate* Hz.

    Uses ffmpeg subprocess so it handles M4A, MP3, FLAC, OGG, etc.
    Falls back to soundfile + torchaudio resample for plain WAV inputs.
    """
    ext = os.path.splitext(src_path)[1].lower()
    if ext == ".wav":
        # Already WAV — read, ensure mono + correct rate, write out
        import torchaudio, torch as _torch
        data, sr = sf.read(src_path, dtype="float32", always_2d=False)
        wav = _torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.T
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sf.write(dst_path, wav.squeeze(0).numpy().astype(np.float32), sample_rate)
    else:
        # Non-WAV: use ffmpeg to decode + resample in one pass
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", str(sample_rate), "-ac", "1", "-f", "wav", dst_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-300:]}")


def text_to_wav(text: str, path: str, sample_rate: int = 24000) -> None:
    """Generate speech for *text* and save to *path* as a mono WAV at *sample_rate* Hz.

    Uses pyttsx3 (cross-platform: SAPI on Windows, espeak on Linux,
    NSSpeechSynthesizer on macOS) for offline synthesis, then resamples to the
    target rate via torchaudio so output is compatible with the moshi/mimi codec.
    """
    import pyttsx3, torchaudio, torch as _torch, tempfile as _tempfile

    # pyttsx3 saves at the system TTS native rate; write to a temp file first
    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
        tmp = _f.name
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, tmp)
        engine.runAndWait()

        # Load with soundfile (works without torchcodec), resample via torchaudio
        data, sr = sf.read(tmp, dtype="float32", always_2d=False)
        wav = _torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        elif wav.ndim == 2:
            wav = wav.T  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sf.write(path, wav.squeeze(0).numpy().astype(np.float32), sample_rate)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def play_wav_file_interruptible(path: str, stop_event: Optional[threading.Event] = None) -> bool:
    """Play a WAV file and optionally stop early when stop_event is set."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; cannot play audio.")
    
    dev_idx = sd.default.device[1]
    logger.info("Playing audio file: %s on output device index: %s", path, dev_idx)
    data, sample_rate = sf.read(path, always_2d=False)
    logger.debug("Read %d samples at %d Hz", len(data), sample_rate)
    
    sd.play(data, samplerate=sample_rate)
    interrupted = False
    
    # Wait for playback to finish
    while True:
        try:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
        except Exception:
            # Fallback if get_stream fails or not supported
            break
            
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            sd.stop()
            logger.info("Playback interrupted.")
            break
        time.sleep(0.02)
    
    logger.info("Playback finished.")
    return interrupted


def _decode_output_tokens(output_text_path: str) -> str:
    if not os.path.exists(output_text_path):
        return ""
    with open(output_text_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return ""
    filtered = [t for t in payload if t not in {"EPAD", "BOS", "EOS", "PAD"}]
    return "".join(filtered).strip()


def _resolve_personaplex_python() -> str:
    if PERSONAPLEX_PYTHON_BIN:
        return PERSONAPLEX_PYTHON_BIN

    # Prioritize the .venv relative to project root
    candidates = [
        Path(sys.prefix) / "Scripts" / "python.exe",
        Path(sys.prefix) / "bin" / "python",
        Path(".venv/Scripts/python.exe"),
        Path(".venv/bin/python"),
        Path("personaplex/.venv/Scripts/python.exe"),
        Path("personaplex/.venv/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _load_env_if_present() -> None:
    if load_dotenv:
        load_dotenv()


async def run_personaplex_offline(
    input_wav: str,
    output_wav: str,
    *,
    output_text: Optional[str] = None,
    voice_prompt: str = PERSONAPLEX_VOICE_PROMPT,
    text_prompt: str = PERSONAPLEX_TEXT_PROMPT,
    voice_prompt_dir: str = PERSONAPLEX_VOICE_PROMPT_DIR,
    device: str = PERSONAPLEX_DEVICE,
    cpu_offload: bool = PERSONAPLEX_CPU_OFFLOAD,
    timeout_seconds: int = PERSONAPLEX_OFFLINE_TIMEOUT_SECONDS,
    seed: int = 42424242,
) -> Dict[str, Any]:
    """Run PersonaPlex offline inference through moshi.offline CLI."""
    _load_env_if_present()
    if output_text is None:
        output_text = os.path.join(tempfile.gettempdir(), "personaplex_output_text.json")

    # Resolve voice prompt path
    final_voice_path_str = _ensure_voice_prompt_exists(voice_prompt)

    command = [
        _resolve_personaplex_python(),
        "-m",
        "moshi.offline",
        "--input-wav",
        input_wav,
        "--output-wav",
        output_wav,
        "--output-text",
        output_text,
        "--voice-prompt",
        final_voice_path_str,
        "--text-prompt",
        text_prompt,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if cpu_offload:
        command.append("--cpu-offload")

    # Attempt direct in-process inference if available
    if moshi_run_inference is not None:
        logger.info("Running PersonaPlex offline inference directly (in-process).")
        try:
            await asyncio.to_thread(
                moshi_run_inference,
                input_wav=input_wav,
                output_wav=output_wav,
                output_text=output_text,
                text_prompt=text_prompt,
                voice_prompt_path=final_voice_path_str,
                tokenizer_path=None,
                moshi_weight=None,
                mimi_weight=None,
                hf_repo="nvidia/personaplex-7b-v1",
                device=device,
                seed=seed,
                temp_audio=0.0, # Default greedy
                temp_text=0.0,
                topk_audio=1,
                topk_text=1,
                greedy=True,
                save_voice_prompt_embeddings=False,
                cpu_offload=cpu_offload
            )
            generated_text = _decode_output_tokens(output_text)
            return {
                "output_wav": output_wav,
                "output_text_path": output_text,
                "generated_text": generated_text,
                "stdout": "in-process execution",
            }
        except Exception as e:
            logger.warning("Direct PersonaPlex inference failed: %s. Falling back to subprocess.", e)

    logger.info("Running PersonaPlex offline inference via subprocess.")
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        logger.error("PersonaPlex offline inference failed: %s", result.stderr)
        raise RuntimeError(
            "PersonaPlex offline inference failed. "
            f"Exit code: {result.returncode}. Stderr: {result.stderr.strip()}"
        )

    generated_text = _decode_output_tokens(output_text)
    return {
        "output_wav": output_wav,
        "output_text_path": output_text,
        "generated_text": generated_text,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def transcribe_audio(audio_waveform: np.ndarray) -> str:
    """Placeholder for audio transcription. Integrate Whisper or similar here."""
    return "[transcription placeholder]"


def extract_text_features(text: str) -> Dict[str, Any]:
    """Extracts features from text."""
    return {"length": len(text)}


def extract_audio_features(audio_waveform: np.ndarray) -> Dict[str, Any]:
    """Extracts features from audio."""
    return {"rms": float(np.sqrt(np.mean(np.square(audio_waveform.astype(np.float32)))))}


def capture_microphone_chunk(duration: float = VOICE_CHUNK_SECONDS, sample_rate: int = VOICE_SAMPLE_RATE) -> np.ndarray:
    """Captures a chunk of audio from the microphone."""
    if sd is None:
        return np.zeros(int(duration * sample_rate), dtype=np.float32)
    chunk = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return chunk.squeeze()


def capture_screen() -> Image.Image:
    """Captures the current screen and returns it as a PIL Image."""
    with mss() as sct:
        # The monitor part can be expanded to support multiple monitors
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        return img
