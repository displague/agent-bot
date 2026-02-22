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
- **Context Management**: Explicitly alternate 'user' and 'assistant' roles in `llm_context` to comply with Gemma-3n constraints.
- **State Tracking**: Monitor `manual_wake` status and "Next event" time-tracking for autonomous operations.

## Guidelines

- **Code Quality**: Follow Python best practices; add docstrings and comments. Use async/await for concurrency.
- **Security**: Protect HF_TOKEN and avoid logging sensitive data.
- **Performance**: Monitor VRAM and offload behavior; `gpt-oss` may offload to disk (`hf_offload/`) and contend with voice inference on constrained memory.
- **Testing**: Validate changes with unit tests (pytest); manual checks for audio/TUI.
- **Collaboration**: Propose changes via pull requests; review for integration impact.
- **Documentation**: Update this file and README.md as features evolve.

## PersonaPlex Specifics

- **Manager**: Use `PersonaPlexManager` for in-process inference. Avoid subprocess calls unless debugging.
- **Voices**: Default to `NATF2.pt` for natural conversational tone.
- **Modes**: Full-duplex streaming is the primary runtime path. Audio is processed in real-time chunks via the `AudioMultiplexer`.

## Project Skills

- **voice-loop-debug** (`.github/skills/voice-loop-debug/SKILL.md`): Triage streaming latency, VAD sensitivity, and in-process inference failures.
- **model-load-triage** (`.github/skills/model-load-triage/SKILL.md`): Monitor HF blob downloads and VRAM offload behavior for Gemma-3n.
- **shutdown-recovery** (`.github/skills/shutdown-recovery/SKILL.md`): Handle the 10s watchdog triggers and ensure the capture thread terminates.
- **env-cuda-alignment** (`.github/skills/env-cuda-alignment/SKILL.md`): Verify bfloat16 compatibility and `moshi` package visibility.
- **multi-modal-diagnose**: Verify `AutoProcessor` state and audio-tool feedback loops.

Agents should operate autonomously while coordinating for seamless development.
