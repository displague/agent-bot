---
name: model-load-triage
description: Debug long model startup and inference failures in agent-bot for transformers or llama_cpp backends, including Hugging Face cache fetch, offload-folder errors, encoding issues, and diagnose timeouts.
---

# Model Load Triage

Use this skill when startup stays on "Loading model... please wait" or inference fails.

## Quick Checks

1. Confirm selected backend and alias:
   - `/model`
   - `/llm-status`
2. Run targeted health check:
   - `/llm-diagnose`
3. Note exact error class:
   - encoding (`charmap` decode)
   - offload folder/device map
   - timeout
   - backend init failure

## Startup Stage Isolation

1. Distinguish **fetching** vs **loading**:
   - Fetching: HF cache/xet network+disk throughput active.
   - Loading: network mostly idle, CPU/GPU and RAM rise while weights map.
2. For fetching suspicion, inspect cache growth and xet logs.
3. For loading suspicion, inspect offload folder and device map behavior.

## Known Remedies

- Enforce UTF-8 runtime on Windows before heavy tokenizer/model I/O.
- Provide `offload_folder` for models that require disk offload re-save.
- Keep diagnose timeout bounded and report elapsed seconds clearly.
- Prefer defaults that are realistic for local hardware.

## PersonaPlex / Moshi Specific

### `step_system_prompts` KV Cache Save/Restore (critical for latency)

`step_system_prompts()` processes the voice identity prompt through the transformer (~46s). It must only run once. After it completes:

1. Call `mimi.reset_streaming()` (matches `offline.py` line 255 — without this, wrong audio codes → silent output)
2. Save lm_gen state to **CPU RAM** via `_save_primed_state()`:
   - VRAM on RTX 5080 Laptop: 15.59GB (lm) + 0.73GB (mimi) = 16.32GB ≈ 100% full
   - `copy.deepcopy()` on GPU → immediate CUDA OOM; must move tensors to CPU first
   - Use `_streaming_state_to_cpu()` to recursively traverse nested dataclass streaming states

3. Before each `tts_stream`/`infer_stream`/`infer` call, restore via `_restore_primed_state()`:
   - Clone CPU tensors back to GPU
   - Call `mimi.reset_streaming()` + `other_mimi.reset_streaming()`
   - Cost: ~1s (tensor clone) vs ~46s (step_system_prompts)

### `_ensure_voice_prompt_exists` Path Walk

This function called `os.walk(Path("."))` on every call — traverses entire project including `.venv/` (thousands of files) and `personaplex/` (large model weights). This blocked the thread for ~46s before OS disk cache warmed. Fix: module-level `_voice_prompt_cache` dict + search `voices/` directory first.

### New LMGen AssertionError

`streaming_forever(1)` in `load()` permanently sets `depformer.is_streaming=True`. Creating a new `LMGen` that wraps the already-streaming model then calling `step_system_prompts` hits an `assert not depformer.is_streaming`. Always reuse `manager.lm_gen` directly — do not create a new `LMGen` after load.

## File Touchpoints

- `main.py`
- `llama_model_manager.py`
- `config.py`
- `utils.py` (PersonaPlexManager — `_save_primed_state`, `_restore_primed_state`, `_streaming_state_to_cpu`)
- `README.md`
