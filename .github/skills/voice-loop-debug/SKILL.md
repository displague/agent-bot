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

- **Thread-Safe Queue Updates**: Microphone capture runs in a background thread. Always use `loop.call_soon_threadsafe(queue.put_nowait, chunk)` to avoid corrupting the async state.
- **Contiguous Arrays**: Slicing numpy arrays often creates non-contiguous views. Use `np.ascontiguousarray(chunk)` before calling `torch.from_numpy()` to prevent `ValueError`.
- **Role Alternation**: Gemma-3n strictly requires alternating `user`/`assistant` roles. Merge consecutive entries from the same role.
- **Warm Generator**: Always use the persistent `self.lm_gen` in `PersonaPlexManager` to avoid the 30s re-instantiation delay.
- **VAD Muting**: Feeding silence (`np.zeros`) to Moshi when the user isn't speaking prevents background noise from poisoning the KV cache.

## Triage Flow

1. **Verify Hardware**: Run `/voice-test-tone`.
   - *Hear a beep?* Output driver is OK.
   - *No beep?* Run `/voice-devices` and check routing. Use `/voice-set-device out <id>`.
2. **Verify Generation**: Check `logs/app.log` for `saved generated audio to ...`.
   - *File exists?* Inference is working.
   - *No file?* Check for `IndexError` (Streaming propagation bug) or OOM.
3. **Verify Hot-Socket**: Connect to port 9999 and run `{"type": "state"}` to check `is_processing` and `voice_activity_state`.
4. **Iterate**: Update logic and run `/logic-reload` via the socket or TUI.

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
