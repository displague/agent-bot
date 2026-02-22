---
name: vram-multi-modal-optimize
description: Strategies for managing VRAM and optimizing multi-modal operations in agent-bot. Use when encountering CUDA out-of-memory errors or slow multi-modal inference.
---

# VRAM and Multi-Modal Optimization

Use this skill when the app crashes with `CUDA error: out of memory` or `Offset increment outside graph capture`.

## Memory Management Strategies

1. **4-Bit Quantization**:
   - Ensure `BitsAndBytesConfig` is utilized in `llama_model_manager.py`.
   - Verify `load_in_4bit=True` is active for the `transformers` backend.

2. **Warm Generator Pattern**:
   - Reuse the `self.lm_gen` instance in `PersonaPlexManager` to avoid frequent OOMs during re-allocation.
   - Use `reset_streaming()` to clear the KV cache between turns.

3. **Memory Cleanup**:
   - Use `torch.cuda.empty_cache()` before starting heavy generation phases.
   - Monitor the startup telemetry for VRAM impact per component.

## Graph Capture Stability

1. **Prefer Eager Mode**:
   - Set `PERSONAPLEX_OPTIMIZE = "eager"` if encountering `AssertionError` or `IndexError`.
   - Eager mode is currently the most stable path for Windows + Moshi.

2. **Idempotent Patching**:
   - Always check `_is_patched` before applying monkeypatches to avoid recursion.

## Diagnostics

- Run `/llm-status` to check the current busy state.
- Check `logs/app.log` for VRAM telemetry: `Weights loaded in X.Xs (VRAM: Y.YGB, +Z.ZGB)`.
