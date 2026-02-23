# Agent-Bot TODO List

This document tracks discussed but unimplemented features and improvements.

## High Priority: Audio & Flow Stability
- [ ] **Phase 3 from Audio Latency Plan**: Implement and test `/voice-set-device <in|out> <id>` command fully in real usage to fix routing.
- [ ] **Jitter Investigation**: Check if `sd.play(chunk, blocking=False)` plus `asyncio.sleep` is the smoothest path or if a custom `sd.OutputStream` callback is required for chunked streaming.
- [ ] **Mimi Bitrate/Quality**: Investigate if we can tune the Mimi encoder/decoder settings for better clarity.

## Phase 12: Memory & Resource Orchestration
- [ ] **Dynamic Gemma Offloading**: Implement a mechanism to move Gemma-3n weights to CPU memory when the agent is in "Reflexive Only" or "Sleeping" modes.
- [ ] **VRAM Sentinel**: Background task to monitor `torch.cuda.memory_allocated()` and alert via TUI if approaching OOM.
- [ ] **Automatic `empty_cache()`**: Trigger periodically during idle periods.

## Phase 13: Semantic Memory (Vector RAG)
- [ ] **Lightweight Vector Store**: Integrate `FAISS` or a numpy-based cosine similarity engine.
- [ ] **Embedding Service**: Use a small local model (e.g., `all-MiniLM-L6-v2`) to index interaction logs.
- [ ] **Context Injection**: Retrieve relevant past interactions and inject them into the `functional_agent` planning phase.

## Phase 10: Audio Output Transformation (DSP Path)
- [ ] **Voice Filter Pipeline**: Insert a processing step between `PersonaPlexManager` and `sounddevice`.
- [ ] **DSP Tools**:
    - [ ] Pitch shifting (numpy/scipy).
    - [ ] Bitcrusher / Robotic effect.
    - [ ] Low-pass / High-pass filters.
- [ ] **Tool Integration**: Allow Gemma-3n to trigger these filters via a `set_voice_filter` tool.

## Phase 15: Visual Sensory Expansion
- [ ] **Region of Interest (ROI)**: Allow `inspect_current_screen` to focus on a specific window or coordinate range.
- [ ] **Webcam Integration**: Add a tool to capture and analyze webcam frames for real-world environmental awareness.

## Developer Experience (DX) Improvements
- [ ] **Socket Client Utility**: Create a standalone `debug_client.py` script for easier hot-socket interaction.
- [ ] **Hot-Reload Enhancements**: Ensure `importlib.reload` correctly handles all edge cases in `interaction_processor`.
- [ ] **Telemetry Dashboard**: Add a TUI screen dedicated to live VRAM and inference timing graphs.
