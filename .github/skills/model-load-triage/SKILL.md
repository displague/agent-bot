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

## File Touchpoints

- `main.py`
- `llama_model_manager.py`
- `config.py`
- `README.md`
