---
name: env-cuda-alignment
description: Align agent-bot runtime with a single .venv and verify CUDA-capable torch stack for PersonaPlex offline voice and local LLM backends. Use when python path, torch build, or CUDA visibility is inconsistent.
---

# Env and CUDA Alignment

Use this skill when diagnostics show wrong python env or `torch` lacks CUDA.

## Current Known-Good Baseline (as of 2026-02-22)

- **Python**: `agent-bot/.venv/Scripts/python.exe`
- **Torch**: `2.10.0+cu130`
- **CUDA**: available, driver 13.0, RTX 5080 Laptop 15.9 GB
- **triton-windows**: `3.6.0`
- **PersonaPlex backend**: fully in-process via `PersonaPlexManager` — no subprocess, no separate venv

> ⚠️ `personaplex/.venv` exists on disk but is **legacy/unused**. It was from the old
> Moshi web-process era (pre-Phase 11). Do NOT treat it as a candidate environment.
> `_resolve_personaplex_python()` in `utils.py` correctly picks `.venv` first via
> `sys.prefix`, but this function is only used by the `run_personaplex_offline` subprocess
> fallback which is itself rarely triggered.

## Baseline Checks

1. From the project root, run:
   ```
   .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   Expected: `2.10.0+cu130 True`
2. Confirm `triton-windows` is present: `uv pip show triton-windows`
3. Confirm app is launched via `uv run main.py` (uses `.venv` automatically).

## Alignment Procedure (if broken)

1. All dependencies live in `agent-bot/.venv`. Install/fix with `uv`.
2. If CUDA missing, install the correct torch+cu wheel for the local driver.
3. Do NOT install into or reference `personaplex/.venv`.
4. Re-run baseline checks to confirm.

## Common Failure Patterns

- CPU-only torch wheel installed after CUDA wheel (check `uv pip show torch`).
- Wrong python used at launch (always use `uv run` or `.venv\Scripts\python.exe`).

## File Touchpoints

- `utils.py` (`PersonaPlexManager`, `_resolve_personaplex_python`)
- `config.py` (`PERSONAPLEX_PYTHON_BIN`, `PERSONAPLEX_DEVICE`)
- `requirements.txt`
