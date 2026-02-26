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
- **Streaming Session Must Reset After TTS**: `tts_stream()` and `infer()` modify `lm_gen` state. After `say_audio()` completes, set `_streaming_session = None` so the next session calls `_restore_primed_state()` for a clean state. Without this, the session runs on corrupted KV cache.
- **Drain Audio Queue After TTS**: While the GPU is locked by TTS generation, audio chunks pile up in `_audio_queue`. After playback, drain all queued chunks so the model doesn't respond to minutes-old microphone audio.
- **TTS Lock Interaction**: `tts_stream`, `infer`, and `infer_stream` all hold `manager._lock` for their full duration. `PersonaPlexStreamingSession.step()` also acquires this lock per 80ms frame. The session is effectively paused during TTS — the `not self._speaking` guard in `_listen_loop` prevents step() calls, but audio keeps accumulating.
- **Session Start Cost**: `PersonaPlexStreamingSession.start()` should use `_restore_primed_state()` (instant) not `step_system_prompts` (~46s). After each TTS call, session is invalidated (`_streaming_session = None`) and recreated on the next voice frame — the fast restore path is critical for responsiveness.
- **Chunked vs Continuous Playback**: Playing each 80ms PCM frame as a separate `play_wav_file_interruptible` call causes ~380ms gaps between syllables. Pre-concatenate all TTS frames into a single buffer and play once via `say_audio()`.
- **Teacher-Forced TTS**: Call `step(text_token=tok, input_tokens=sine_frame)` WITHOUT `moshi_tokens` — depformer samples audio autoregressively. Passing `moshi_tokens=zero_frame` forces silent PAD tokens (see issue #X). The first 1-2 frames are quiet; amplitude appears from frame 3+.
- **Greedy Decoding = Stuck Output**: LMGen initialized with `top_k=1, top_k_text=1` produces fully deterministic output. Combined with `_restore_primed_state()` restoring the same KV-cache state every call, the model generates the identical audio every time (always the startup greeting). MLX reference uses `audio_topk=250, text_topk=25`. Match these defaults.
- **Mimi Routing (Bug A+B)**: In `step()`/`infer()`/`infer_stream()`, decode with `mimi.decode(tokens[:,1:9])` (NOT `other_mimi.decode`). Also call `_ = other_mimi.decode(...)` AND `_ = other_mimi.encode(chunk)` as discards for state sync every frame — skipping state sync causes codec drift and degraded audio.

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
