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
2. Capture key fields from **Esc** debug screen:
   - `Inf` (inference ms/frame): real-time budget is **80ms**.
   - `VRAM`: usage above 16GB indicates system RAM swapping/slowness.
   - `Tokens`: incoherent output ("word salad") indicates slowness or high temp.
3. Verify Esc debug view is populated with `Phase` and `Voice` state.

## Common Pitfalls & Lessons

- **Zero-Latency Streaming**: Always use `say_stream()` for agent speech. This bridges the synchronous `tts_stream()` generator to the async `_playback_queue`, ensuring speech starts <1s.
- **KV-Cache Reset**: Every conversational turn must start with an explicit `lm_gen.reset_streaming()` before calling `_restore_primed_state()`. Failure to do this causes the model to repeat previous turns (e.g., the startup greeting).
- **Greedy Text Decoding**: To resolve incoherent "word salad" output at slow inference speeds, use `temp_text=0.0` and `top_k_text=1`.
- **Thread-Safe Queue Updates**: Microphone capture runs in a background thread. Always use `loop.call_soon_threadsafe(queue.put_nowait, chunk)` to avoid corrupting the async state.
- **Contiguous Arrays**: Slicing numpy arrays often creates non-contiguous views. Use `np.ascontiguousarray(chunk)` before calling `torch.from_numpy()` to prevent `ValueError`.
- **Warm Generator**: Always use the persistent `self.lm_gen` in `PersonaPlexManager` to avoid the 30s re-instantiation delay.
- **VAD Muting**: Feeding silence (`np.zeros`) to Moshi when the user isn't speaking prevents background noise from poisoning the KV cache.
- **Streaming Session Must Reset After TTS**: `tts_stream()` and `infer()` modify `lm_gen` state. After `say_audio()` or `say_stream()` completes, set `_streaming_session = None` so the next session starts fresh.
- **Mimi Routing (Bug A+B)**: In `step()`/`infer()`/`infer_stream()`, decode with `mimi.decode(tokens[:, 1:lm.dep_q+1])`. Also call discard state syncs for `other_mimi` every frame (`other_mimi.encode` and `other_mimi.decode`) to prevent codec drift.
- **CPU-First Weight Loading**: To avoid VRAM OOM on 16GB GPUs, load weights into system RAM first (`map_location='cpu'`). Quantize on CPU then move the reduced model to GPU.

## Triage Flow

1. **Verify Hardware**: Run `/voice-test-tone`.
   - *Hear a beep?* Output driver is OK.
   - *No beep?* Run `/voice-devices` and check routing. Use `/voice-set-device out <id>`.
2. **Verify Generation**: Check `logs/app.log` for `Voice model (streaming): ...`.
   - *Text appears?* Inference is working.
   - *Coherent?* If not, check `Inf` latency on Esc screen.
3. **Verify Hot-Socket**: Connect to port 9999 and run `{"type": "state"}` to check `vram_gb` and `inference_ms`.
4. **Iterate**: Update logic and run `/logic-reload` via the socket or TUI.

## File Touchpoints

- `voice_loop.py`
- `utils.py` (PersonaPlexManager, AudioMultiplexer)
- `interaction_processor.py`
- `quantize.py`

## Fix Patterns

- Add bounded timeouts for inference and shutdown waits.
- Ensure queued items are always drained or re-queued on failure.
- Keep diagnostics commands lightweight and non-blocking.
- Prefer explicit status messages over implicit spinner-only states.
