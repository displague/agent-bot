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

2. **CPU Offloading**:
   - Enable `PERSONAPLEX_CPU_OFFLOAD = True` in `config.py`.
   - This moves PersonaPlex layers to CPU, freeing critical VRAM for the primary LLM.

3. **Memory Cleanup**:
   - Use `torch.cuda.empty_cache()` before starting heavy generation phases.
   - Monitor the startup telemetry for VRAM impact per component.

## Graph Capture Stability

1. **Disable CUDA Graphs**:
   - Set `PERSONAPLEX_USE_CUDA_GRAPHS = False` if encountering `Offset increment` errors.
   - Graphs improve speed but can be brittle on Windows or with dynamic input shapes.

2. **Serialization**:
   - Always use the `_processing_lock` in `InteractionProcessor` and `ThoughtGenerator`.
   - Concurrent model calls are the leading cause of graph capture failures.

## Diagnostics

- Run `/llm-status` to check the current busy state.
- Check `logs/app.log` for VRAM telemetry: `Weights loaded in X.Xs (VRAM: Y.YGB, +Z.ZGB)`.
