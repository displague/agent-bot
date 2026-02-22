# Agent Collaboration Guidelines for Agent-Bot

This document outlines the roles, tools, protocols, and guidelines for AI agents collaborating on the development and maintenance of the Agent-Bot project. The Agent-Bot is an autonomous AI system with terminal-first voice capabilities via NVIDIA PersonaPlex offline mode, multi-phase reasoning, dynamic model switching, and a TUI interface.

## Roles

- **Audio Processing Agent**: Handles PersonaPlex integration for speech-to-speech conversion, voice conditioning, and persona prompts. Manages audio input/output, Opus codec usage, and offline continuous voice mode.
- **Reasoning Agent**: Oversees multi-phase processing (planning, execution, digestion, validation, response) using the model manager. Simulates function calls for tasks like searching indexes or scheduling events.
- **UI Agent**: Manages the curses-based TUI, including rendering, input handling (text/voice), scrolling, and screen switching. Integrates audio controls with the interface.
- **Logging and Indexing Agent**: Maintains interaction logs (JSONL), compressed summaries, and keyword-based indexes. Handles log compression, context retrieval, and state tracking.
- **Event Management Agent**: Manages the async event queue for scheduling, reminders, lookups, RAG completions, and training events.
- **Autonomous Operation Agent**: Generates periodic thoughts during active hours (7 AM - 11 PM), monitors sleep status, and triggers autonomous actions.
- **Integration Agent**: Coordinates overall system integration, dependency management, configuration updates, and testing.

## Tools Available

- **Model Inference**: Configurable backends: `llama_cpp` GGUF models and `transformers` models (e.g., `gpt-oss`) from HF cache; PersonaPlex (via moshi) for audio processing.
- **Logging**: Async logging to files and stderr; in-memory log management with locking.
- **Indexing**: Keyword-based search and context retrieval from interaction history.
- **Event Scheduling**: Async queue for deferred tasks and timed events.
- **Audio Handling**: Speech-to-speech via PersonaPlex offline mode with continuous loop support; Opus for compression.
- **TUI Controls**: Curses-based interface for real-time display and input.
- **File Management**: JSONL logs, compressed logs, index files, configuration constants.
- **Environment**: `.env` for optional `HF_TOKEN`; HF cache for model artifacts; PATH for Opus binaries; GPU monitoring.

## Interaction Protocols

- **Communication**: Use structured messages with role prefixes (e.g., "[Audio Agent] Processing voice input"). Reference file paths and line numbers for code changes.
- **Task Breakdown**: Decompose complex tasks into phases: Discovery (research), Alignment (plan), Design (draft), Refinement (revise), Implementation (code), Verification (test).
- **Commit Protocol**: Commit frequently with detailed messages (e.g., "Integrate PersonaPlex STT in utils.py"). Push to branch `multiple_files` for collaboration.
- **Error Handling**: Log errors with context; retry failed operations up to 3 times; escalate to human if critical.
- **Autonomous Mode**: During active hours, generate thoughts independently; respect sleep hours (11 PM - 7 AM) for minimal activity.
- **PersonaPlex Integration**: Use voice prompts (e.g., NATF2.pt) and text prompts for consistent personas. Default runtime uses `agent-bot/.venv` Python and offline continuous voice mode.
- **Model Controls**: Use `/model`, `/model list`, `/model use <alias>` for runtime model operations; keep `config.py` model aliases current.
- **Runtime Diagnostics**: Use `/llm-status`, `/llm-diagnose`, and `/voice-diagnose` for fast health checks. Use `/force-quit` or second `Ctrl+C` to hard-stop stuck shutdowns.
- **State Tracking**: Maintain awareness of unprocessed interactions, ongoing thoughts, and next events. Update state after each action.

## Guidelines

- **Code Quality**: Follow Python best practices; add docstrings and comments. Use async/await for concurrency.
- **Security**: Protect HF_TOKEN and avoid logging sensitive data.
- **Performance**: Monitor VRAM and offload behavior; `gpt-oss` may offload to disk (`hf_offload/`) and contend with voice inference on constrained memory.
- **Testing**: Validate changes with unit tests (pytest); manual checks for audio/TUI.
- **Collaboration**: Propose changes via pull requests; review for integration impact.
- **Documentation**: Update this file and README.md as features evolve.

## PersonaPlex Specifics

- **Voices**: Use NAT (natural) or VAR (variety) embeddings for consistency.
- **Prompts**: Text-based for roles (e.g., "You are a wise assistant."); audio-based for voice conditioning.
- **Modes**: Offline continuous mode is the primary runtime path for terminal operation.
- **Dependencies**: Ensure moshi installed; Opus codec available.

Agents should operate autonomously while coordinating for seamless development.
