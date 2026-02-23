---
name: multi-modal-diagnose
description: Verify multi-modal model state, AutoProcessor availability, and diagnose audio-tool feedback loops in Gemma-3n. Use when the model fails to 'hear' or incorrectly processes auditory sensory context.
---

# Multi-Modal Diagnose

Use this skill when Gemma-3n output indicates it cannot access or reason about audio context.

## Baseline Checks

1. Confirm `hf_processor` is initialized in `llama_model_manager.py`.
2. Check logs for "Tool returned raw audio, re-invoking LLM...".
3. Verify `inspect_audio_snippet` is being called via `/llm-status` or debug phase visibility.

## Triage Flow

1. **Processor Check**:
   - Run `/llm-diagnose` and ensure no `ImportError` or `AttributeError` related to `AutoProcessor`.
   - Confirm `trust_remote_code=True` was used during load.

2. **Tool Execution**:
   - Submit interaction: "Inspect the last 5 seconds of audio."
   - Observe debug phase: Should move through Planning -> Execution.
   - Confirm tool output in logs: Should show "Audio Snippet (5.0s) Features: {...}" or raw audio broadcast.

3. **Feedback Loop**:
   - If `return_raw=True`, ensure the second LLM invocation triggers.
   - Verify input sampling rate matches Gemma expectations (usually 16kHz).

## File Touchpoints

- `llama_model_manager.py`
- `functional_agent.py`
- `utils.py` (RollingAudioBuffer)

## Fix Patterns

- Ensure `torch.is_floating_point(v)` is used before `bfloat16` conversion.
- Verify `apply_chat_template` includes the follow-up prompt for retrieved audio.
- Check `RollingAudioBuffer` queue for stale or empty chunks.
