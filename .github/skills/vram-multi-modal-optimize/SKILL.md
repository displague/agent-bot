---
name: vram-multi-modal-optimize
description: Strategies for managing VRAM and optimizing multi-modal operations in agent-bot. Use when encountering CUDA out-of-memory errors or slow multi-modal inference.
---

# VRAM and Multi-Modal Optimization

Use this skill when the app crashes with `CUDA error: out of memory` or `Offset increment outside graph capture`.

## VRAM Budget (RTX 5080 Laptop, 16 GB)

- PersonaPlex LM: **15.59 GB**
- Mimi codec: **0.73 GB**
- Total: **16.32 GB ≈ 100% full at all times**
- Do NOT introduce extra model copies or deepcopy on GPU (immediate OOM).

## Quantization Reality for Moshi

**Post-load PyTorch quantization (`--quantize 4bit`) yields only ~0.21 GB VRAM savings — not the expected ~7 GB.**

Root cause: All large Moshi weight matrices live in fused-ops layers (`gating`, `linear_in`, `linear_out`, `out_proj`) that access `.weight` directly and **must be skipped** during quantization. Only small misc layers get quantized.

**The correct path for real VRAM reduction:**
- The MLX reference project (`external/personaplex-mlx`) downloads **pre-quantized model files** from HuggingFace (`nvidia/personaplex-7b-v1/model.q4.safetensors`). The weights are stored quantized on disk — not post-load quantized.
- For PyTorch: `bitsandbytes` (4-bit NF4/INT8) or GGUF format would be needed.
- Track progress in GH #8 (post-load ineffectiveness) and #9 (pre-quantized HF file research).

## Memory Management Strategies

1. **Warm Generator Pattern**:
   - Reuse the `self.lm_gen` instance in `PersonaPlexManager` to avoid frequent OOMs during re-allocation.
   - Use `_restore_primed_state()` (not `reset_streaming()`) to clear state between turns — this clones saved CPU tensors back to GPU.

2. **torch.no_grad()**:
   - All 5 inference lock sections in `utils.py` are wrapped in `torch.no_grad()`. Keep it that way.

3. **Memory Cleanup**:
   - Use `torch.cuda.empty_cache()` before starting heavy generation phases.
   - Monitor the startup telemetry for VRAM impact per component.

## Graph Capture Stability

1. **Prefer Eager Mode**:
   - Set `PERSONAPLEX_OPTIMIZE = "eager"` if encountering `AssertionError` or `IndexError`.
   - Eager mode is currently the most stable path for Windows + Moshi.

2. **Idempotent Patching**:
   - Always check `_is_patched` before applying monkeypatches to avoid recursion.

## Inference Speed

Current speed: ~500ms/frame. Real-time budget: 80ms/frame. We are ~6x too slow.

- Bottleneck is full-precision 7B inference on maxed VRAM.
- Real quantization (bitsandbytes, GGUF, or pre-quantized HF files) is required to improve speed.
- `torch.compile` / CUDA graphs would help once VRAM is freed.

## Diagnostics

- Run `/llm-status` to check the current busy state.
- Run `/voice-status` to see VRAM and PersonaPlex state.
- Check `logs/app.log` for VRAM telemetry: `Weights loaded in X.Xs (VRAM: Y.YGB, +Z.ZGB)`.

