---
name: voice-loop-debug
description: Diagnose offline continuous voice loop issues in agent-bot, especially when inputs enqueue but no spoken or text response appears. Use for voice activity mismatches, silent listening states, and Esc debug-screen validation.
---

# Voice Loop Debug

Use this skill when the app shows voice loop running but user sees no response or transcription.

## Quick Checks

1. Confirm runtime status from UI commands:
   - `/voice-status`
   - `/voice-diagnose` (ensure `moshi` is available)
   - `/llm-status`
2. Capture key fields:
   - voice mode/activity (`listening`, `Model: ...` thinking state)
   - `unprocessed` count trend
   - llm `processing` flag and `processing_phase`
3. Verify Esc debug view is populated with `phase` and `voice` state.

## Common Pitfalls & Lessons

- **Thread-Safe Queue Updates**: Microphone capture runs in a background thread. Always use `loop.call_soon_threadsafe(q.put_nowait, chunk)` to avoid corrupting the async state of subscriber queues.
- **Contiguous Arrays**: Slicing numpy arrays often creates non-contiguous views. Use `np.ascontiguousarray(chunk)` before calling `torch.from_numpy()` to prevent `ValueError` or `AssertionError` during inference.
- **Role Alternation**: Gemma-3n strictly requires alternating `user`/`assistant` roles. If the log schema is unified, ensure consecutive entries from the same role are merged before templating.

## Triage Flow

1. If voice loop is `running` but no transcription appears:
   - Check `AudioMultiplexer` log for "starting capture thread".
   - Confirm `StreamingSession` is created and `start()` completes.
2. If state stalls on `processing utterance`:
   - Inspect `PersonaPlexManager` logs for in-process inference completion.
   - Verify `playback_interrupt` isn't stuck.
3. If `voice loop activity=error`:
   - Surface the full stderr cause.
   - Check if `subprocess` fallback was triggered and if it timed out.

## File Touchpoints

- `voice_loop.py`
- `utils.py` (PersonaPlexManager, AudioMultiplexer)
- `interaction_processor.py`

## Fix Patterns

- Add bounded timeouts for inference and shutdown waits.
- Ensure queued items are always drained or re-queued on failure.
- Keep diagnostics commands lightweight and non-blocking.
- Prefer explicit status messages over implicit spinner-only states.
