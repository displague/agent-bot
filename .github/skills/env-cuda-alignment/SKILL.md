---
name: env-cuda-alignment
description: Align agent-bot runtime with a single .venv and verify CUDA-capable torch stack for PersonaPlex offline voice and local LLM backends. Use when python path, torch build, or CUDA visibility is inconsistent.
---

# Env and CUDA Alignment

Use this skill when diagnostics show wrong python env or `torch` lacks CUDA.

## Baseline Checks

1. Run `/voice-diagnose` and capture:
   - python path used by PersonaPlex
   - torch version
   - `cuda_available`
   - CUDA version
2. Confirm app runtime python and voice runtime python match expected `.venv`.

## Alignment Procedure

1. Standardize to one environment (`agent-bot/.venv`) for app + voice subprocess.
2. Install dependencies with `uv` in that environment.
3. If CUDA missing, install the correct torch/cu wheel set for the local driver/toolkit.
4. Re-run diagnostics and verify CUDA is visible from the same interpreter path.

## Common Failure Patterns

- App using `.venv` while voice subprocess points at stale `personaplex/.venv`.
- CPU-only torch wheel installed after CUDA wheel.
- Mixed package sources causing silent backend fallback.

## File Touchpoints

- `utils.py`
- `requirements.txt`
- `README.md`
- `AGENT.md`
