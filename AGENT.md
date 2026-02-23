# Agent Collaboration Guidelines for Agent-Bot

This document outlines the roles, tools, protocols, and guidelines for AI agents collaborating on the development and maintenance of the Agent-Bot project. The Agent-Bot is an autonomous AI system with terminal-first voice capabilities via NVIDIA PersonaPlex offline mode, multi-phase reasoning, dynamic model switching, and a TUI interface.

## Roles

- **Audio Processing Agent**: Manages the `AudioMultiplexer` and `PersonaPlexManager`. Orchestrates full-duplex streaming, in-process model inference, and reflexive verbal fillers.
- **Sensory Agent**: Maintains the `RollingAudioBuffer` (auditory memory) and provides tools like `inspect_audio_snippet` for multi-modal context analysis.
- **Reasoning Agent**: Oversees multi-phase processing (Notes, Planning, Execution, Digesting, Validating, Responding) using Gemma-3n. Ensures conversation history strictly alternates roles.
- **UI Agent**: Manages the TUI with enhanced navigation: Up/Down for input history, PgUp/PgDn for log scrolling, and Left/Right for cursor editing.
- **Logging Agent**: Ensures a unified log schema via `InteractionLogManager`, facilitating accurate summaries by the `EventCompressor`.

## Tools Available

- **Model Inference**: Gemma-3n (multi-modal transformer) and legacy llama.cpp backends. `PersonaPlexManager` keeps audio models warm in VRAM for rapid response.
- **Auditory Backbone**: `AudioMultiplexer` for broadcast mic capture and `RollingAudioBuffer` for 10s sensory memory.
- **TUI Controls**: Curses-based interface with `/wake`, `/sleep`, and real-time processing phase visibility.
- **Shutdown Watchdog**: 10-second hard-kill watchdog ensuring reliable process termination.

## Interaction Protocols

- **Communication**: Use role prefixes and reference the current processing phase (e.g., "[Reasoning Agent] Phase: Planning").
- **Handoff Logic**: Priority 1: Trigger reflexive filler. Priority 2: Multi-phase background reasoning. Priority 3: Final spoken response.
- **Context Management**: Explicitly alternate 'user' and 'assistant' roles in `llm_context` to comply with Gemma-3n constraints. Merge consecutive entries from the same role.
- **GPU Serialization**: Always acquire `_processing_lock` before making model calls (LLM or PersonaPlex) to prevent CUDA graph conflicts and VRAM spikes.
- **State Tracking**: Monitor `manual_wake` status and "Next event" time-tracking for autonomous operations.

## Guidelines

- **Code Quality**: Follow Python best practices; add docstrings and comments. Use async/await for concurrency.
- **Security**: Protect HF_TOKEN and avoid logging sensitive data.
- **Multi-Modal Optimization**: Prefer 4-bit quantization (`BitsAndBytesConfig`) and CPU offloading for large models to manage VRAM effectively.
- **Thread Safety**: Use `loop.call_soon_threadsafe` when broadcasting from background capture threads to async queues.
- **Testing**: Validate changes with unit tests (pytest). Use the global mocking strategy in `tests/conftest.py` to avoid heavy model loads. To test real logic that is globally mocked, use `@pytest.mark.skip_heavy_mock`. Always set a timeout (e.g., `--timeout=30`) when running tests.
- **Collaboration**: Propose changes via pull requests; review for integration impact.
- **Documentation**: Update this file and README.md as features evolve.

## PersonaPlex Specifics

- **Manager**: Use `PersonaPlexManager` for in-process inference. Avoid subprocess calls unless debugging.
- **Voices**: Default to `NATF2.pt` for natural conversational tone.
- **Modes**: Full-duplex streaming is the primary runtime path. Audio is processed in real-time chunks via the `AudioMultiplexer`.
- **TTS bypass** (`tts_stream`): Teacher-forces text tokens so the depformer samples audio autoregressively — fast, deterministic speech without a conversational exchange.
- **Conversational path** (`hear_stream`): Generates pyttsx3 TTS audio of the "heard" text, feeds it to `infer()`, and yields the model's natural spoken reply. Use this when you want PersonaPlex's genuine conversational voice.
- **Voice primer**: `voices/introduce-yourself.wav` — a 24 kHz WAV of "Please introduce yourself." ready to feed to `hear_stream` or `infer`.

## Debug Commands (port 9999 + TUI)

Send JSON lines to port 9999: `{"type": "command", "data": "/voice-say Hello world"}`

| Command | Description |
|---|---|
| `/voice-say <text>` | Inject text as an `override_response`; spoken via `tts_stream` (bypass path). |
| `/voice-hear <text>` | Synthesise text as user speech via pyttsx3, feed to PersonaPlex conversationally, play agent reply. |
| `/voice-diagnose` | Print python path, torch/CUDA info. |
| `/voice-start` / `/voice-stop` | Start/stop the offline continuous voice loop. |
| `/set-persona <text>` | Update `PERSONAPLEX_TEXT_PROMPT` at runtime. |

## Project Skills

- **voice-loop-debug** (`.github/skills/voice-loop-debug/SKILL.md`): Triage streaming latency, VAD sensitivity, and in-process inference failures.
- **model-load-triage** (`.github/skills/model-load-triage/SKILL.md`): Monitor HF blob downloads and VRAM offload behavior for Gemma-3n.
- **shutdown-recovery** (`.github/skills/shutdown-recovery/SKILL.md`): Handle the 10s watchdog triggers and ensure the capture thread terminates.
- **env-cuda-alignment** (`.github/skills/env-cuda-alignment/SKILL.md`): Verify bfloat16 compatibility and `moshi` package visibility.
- **multi-modal-diagnose**: Verify `AutoProcessor` state and audio-tool feedback loops.
- **vram-multi-modal-optimize**: Strategies for managing memory and graph capture stability.
- **debug-link**: Instructions for connecting to the Port 9999 hot socket.

### **Phase 6: Audio Multiplexer (Clean Stream Broadcast)**
*   **Goal:** Decouple the microphone stream from the `VoiceLoop` logic.
*   **Status:** COMPLETED
*   **Action:** Implemented `AudioMultiplexer` and `RollingAudioBuffer` for thread-safe, multi-subscriber audio capture.

### **Phase 7: True Full-Duplex (Low-Latency Interjections)**
*   **Goal:** Enable the agent to start responding or interject *while* the user is still speaking.
*   **Status:** COMPLETED
*   **Action:** Shifted to a streaming approach where `LMGen.step` is processed in real-time via `PersonaPlexStreamingSession`.

### **Phase 8: Vision Integration (Multi-Modal Sensory)**
*   **Goal:** Allow the agent to "see" its environment.
*   **Status:** COMPLETED
*   **Action:**
    1.  Implemented `capture_screen` utility using `mss`.
    2.  Added `inspect_current_screen` tool in `LlamaModelManager`.
    3.  Enabled multi-modal visual analysis in `llm_call`.
    4.  Updated system prompt to include visual sensory instructions.

### **Phase 11: Performance & Flow Overhaul (The Hub Update)**
*   **Goal:** Eliminate speech latency and enable rapid iteration.
*   **Status:** COMPLETED
*   **Action:**
    1.  Implemented `Warm Generator` architecture (persistent `LMGen`).
    2.  Shifted to `Streaming Playback` (chunks spoke as generated).
    3.  Created `Real-time Experimentation Hub` (Port 9999 socket).
    4.  Implemented `Deep Hot-Reload` (`/logic-reload` swaps code without weight reloads).
    5.  Resolved Moshi `IndexError` and `AssertionError` via monkeypatches.

### **Phase 12: Memory & Resource Orchestration**
*   **Goal:** Manage the 23GB+ VRAM footprint of having both models loaded.
*   **Action:**
    1.  Implement a "sleep mode" for Gemma where its weights are offloaded to CPU/Disk when inactive.
    2.  Add a background task that monitors `torch.cuda.memory_reserved()` and triggers `empty_cache()`.

### **Phase 10: Audio Output Transformation (DSP Path)**
*   **Goal:** Allow the agent to transform its own spoken voice via tools.
*   **Action:**
    1.  Implement a transformable audio output pipeline between PersonaPlex and the sound drivers.
    2.  Add a `set_voice_filter(filter_type, params)` tool.
    3.  Support real-time DSP effects like pitch-shifting (e.g., lower/higher tone) or robotic modulation.
    4.  Ensure the implementation is cross-platform (avoiding Windows-only APIs where possible).

Agents should operate autonomously while coordinating for seamless development.
