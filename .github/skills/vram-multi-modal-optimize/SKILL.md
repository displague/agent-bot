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

Root cause: All large Moshi weight matrices live in fused-ops layers (`gating`, `linear_in`, `linear_out`, `out_proj`) that access `.weight` directly and **must be skipped** during naive quantization. Only small misc layers get quantized.

**The VRAM Spike Blocker:**
Attempting to quantize these layers with a custom `_QuantizedLinear` that dequantizes in the `forward()` pass results in a massive VRAM spike (doubling usage up to 30GB+) because the dequantized weights are held in VRAM during computation.

**The correct path for real VRAM reduction:**
- Switch to **pre-quantized model files** using `bitsandbytes` or GGUF.
- We currently use `model_bnb_4bit.pt` from the `brianmatzelle/personaplex-7b-v1-bnb-4bit` community repo.
- This allows computation to happen directly on the quantized weights using native `bitsandbytes` kernels.

## Memory Management Strategies

1. **CPU-First Weight Loading**:
   - Always load the 14GB payload into system RAM first (`device="cpu"`) via `safetensors` or `torch.load(..., map_location='cpu')`.
   - Quantize on CPU (if needed) then move the reduced model to GPU.
   - Use `CUDA_VISIBLE_DEVICES=""` during load to definitively isolate the GPU.

2. **Warm Generator Pattern**:
   - Reuse the `self.lm_gen` instance in `PersonaPlexManager` to avoid frequent OOMs during re-allocation.
   - Use `_restore_primed_state()` (not `reset_streaming()`) to clear state between turns — this clones saved CPU tensors back to GPU.
   - **Crucial**: Call `lm_gen.reset_streaming()` *before* restoring state to avoid repeated greeting phrases.

3. **Memory Cleanup**:
   - Use `gc.collect()` and `torch.cuda.empty_cache()` before starting heavy weight transfers.
   - Monitor the Esc debug screen for real-time `VRAM` and `Inf` latency.

## Inference Speed

Current speed: ~500ms/frame. Real-time budget: 80ms/frame. We are ~6x too slow.

- Bottleneck is VRAM-to-System-RAM overflow (swapping) when usage exceeds 16GB.
- Real quantization (`bitsandbytes` NF4) is required to keep the model under 10GB and achieve real-time speed.

## Diagnostics

- Run `/llm-status` to check the current busy state.
- Run `/voice-status` to see VRAM and PersonaPlex state.
- Check `logs/app.log` for VRAM telemetry: `Weights loaded in X.Xs (VRAM: Y.YGB, +Z.ZGB)`.

