# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent that can respond to queries, compress historical data, and schedule future actions.

## Features

- **Bidirectional Audio Interaction**: Uses NVIDIA PersonaPlex for real-time, full-duplex speech-to-speech conversations with persona control and voice conditioning.
- **Auditory Memory**: 10-second rolling audio buffer powered by a shared `AudioMultiplexer`, allowing the agent to "hear" and analyze environmental context.
- **Visual Sensory Memory**: Native screen capture using `mss`, allowing the agent to "see" and reason about visual context.
- **Multi-Modal Reasoning**: Powered by `google/gemma-3n-E2B-it`, supporting native text, audio, and vision analysis with 4-bit quantization.
- **Real-Time Telemetry**: VRAM impact and duration tracking for all model loading stages, providing clear visibility into resource usage during startup.
- **Cognitive Handoff**: Immediate reflexive "verbal fillers" via PersonaPlex while the LLM performs deep reasoning in the background.
- **Multi-Phase Reasoning**: Breaks down tasks into planning, execution, digestion, validation, and response phases with real-time UI visibility.
- **Model Switching**: Runtime `/model` command to inspect and switch between configured model backends (default is `gemma-it`).
- **Autonomous Operation**: Generates periodic "thoughts" during active hours, compresses interaction logs hourly, and schedules events like reminders or training.
- **Logging and Indexing**: Unified interaction logging via `InteractionLogManager` for consistent schemas and accurate summarization.
- **Enhanced TUI Interface**: Real-time curses interface with status, thoughts, and interaction history. Features input history (Up/Down), log scrolling (PgUp/PgDn), and cursor editing (Left/Right).
- **Robust Shutdown**: Graceful shutdown sequence with a 10-second hard-kill watchdog.

## Architecture

- `main.py`: Entry point; initializes the auditory backbone, model managers, and runs all components asynchronously.
- `runtime_manager.py`: Manages shared executors and tracked async tasks for graceful shutdown.
- `tui_renderer.py`: Manages the curses-based TUI; handles rendering, input history, cursor editing, and log scrolling.
- `interaction_processor.py`: Orchestrates cognitive handoff between immediate fillers and multi-phase LLM reasoning with a shared processing lock.
- `functional_agent.py`: Manages multi-phase processing and explicitly tracks conversation history (user/assistant roles).
- `llama_model_manager.py`: Manages model backends, including Gemma-3n multi-modal support via `AutoProcessor`, bfloat16 precision, and 4-bit quantization.
- `thought_generator.py`: Generates autonomous thoughts with manual wake/sleep override and synchronized model access.
- `event_scheduler.py`: Tracks and triggers timed events, updating the "Next event" status in real-time.
- `utils.py`: Contains core infrastructure: `AudioMultiplexer`, `RollingAudioBuffer`, `PersonaPlexManager` (persistent VRAM models), and `PersonaPlexStreamingSession`.
- `voice_loop.py`: Full-duplex streaming voice loop that processes audio chunks in real-time via the multiplexer.
- `config.py`: Central configuration for paths, model catalog, timeouts, and verbal fillers.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- CUDA 12.1+ (Tested with CUDA 13.0)
- Opus audio codec: Install `libopus-dev` (Ubuntu/Debian) or equivalent.
- Hugging Face account with access to [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) and [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd agent-bot
   ```

2. Create and use the project virtualenv:
   ```bash
   uv venv .venv
   ```

3. Install Python dependencies from lockfile:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Install PersonaPlex:
   ```bash
   git clone https://github.com/NVIDIA/personaplex.git
   # The requirements.txt already includes the upstream dependency.
   ```

5. Set up environment:
   - Create `.env` file with `HF_TOKEN=<your-huggingface-token>`
   - Ensure Opus binaries (`opusdec`, `opusenc`) are in PATH (e.g., `~/.local/bin`)

6. Model setup:
   - No local `model.bin` is required by default.
   - Models are resolved/downloaded into standard Hugging Face cache locations.
   - Configure aliases in `config.py` (`MODEL_LIST`) and switch at runtime with `/model use <alias>`.
   - PersonaPlex voices: Download embeddings (e.g., NATF2.pt) as needed.

### Running

```bash
python main.py
```

Optional flags:
- `--reset-logs`: Clear all log files and offload caches on startup.
- `--dev`: Enable development mode (disables autonomous thoughts/compression).

Quick first-run validation:
1. Start with simple UI and reset logs:
   ```bash
   # PowerShell
   $env:AGENTBOT_UI_MODE='simple'; python main.py --reset-logs --dev
   ```
2. At the prompt, run:
   - `/help`
   - `/model`
   - `/model list`
   - `/model use gpt-oss`
   - `/smoke`
   - `/smoke-model`
   - or `/smoke-all`
   - `/quit` to exit cleanly
3. Verify evidence in:
   - `logs/smoke_test.jsonl`

Use `Ctrl+R` (or `/voice-start`) in TUI to start offline continuous voice mode.
Use `/voice-stop` to stop voice mode, `/voice-status` to check status, and `/voice-diagnose` to inspect runtime torch/CUDA setup.
Use `/llm-status` for live model processing state and `/llm-diagnose` for quick model health checks.
Use `/force-quit` for immediate hard-stop when graceful shutdown is stuck.
Voice loop auto-starts on launch by default; disable in `config.py` via `VOICE_AUTO_START_ON_LAUNCH = False`.
If curses is unavailable (for example on native Windows Python), the app falls back to a simple stdin mode automatically.
In TUI mode, press `Tab` after a `/` command prefix for autocomplete.
In TUI mode: `Esc` toggles debug view, `Up/Down` scroll log history, and `Backspace` edits input.
In TUI mode: `Ctrl+D` exits. During shutdown, a second `Ctrl+C` force-kills the process tree.

PersonaPlex CLI check:
```bash
.venv/Scripts/python.exe -m moshi.offline --help
```

## Configuration

- `config.py`: Adjust paths, sleep hours (11 PM - 7 AM), worker count (5), and model aliases/catalog.
- Voice/Persona: Configure in `utils.py` or TUI for PersonaPlex prompts.
- `AGENTBOT_UI_MODE`: `auto` (default), `simple`, or `curses`.
- `AGENTBOT_DEV_MODE`: `1` disables autonomous background thought/compression tasks for rapid debugging.
- `AGENTBOT_LOG_LEVEL`: default `INFO`; set to `DEBUG` for verbose diagnostics.
- `AGENTBOT_LOG_PROMPTS`: set to `1` only when you need full prompt dumps for debugging.
- `SHUTDOWN_GRACE_SECONDS`: timeout used for graceful shutdown before force-cancel.
- `AGENTBOT_MODEL_ALIAS`: optional startup model alias override (e.g. `default` or `gpt-oss`).
- `AGENTBOT_LLAMA_N_CTX`: optional llama.cpp context override (default 2048).

Log files:
- `logs/app.log`: rotating diagnostics log.
- `logs/hard_log.jsonl`: structured interaction records only.
- `logs/smoke_test.jsonl`: smoke test evidence.
- `logs/personaplex_server.log`: PersonaPlex server stdout/stderr for voice startup/runtime failures.
- `hf_offload/`: transformer offload directory when running large models with `device_map="auto"`.

## Development

- Commit often with detailed messages.
- See [AGENT.md](AGENT.md) for collaboration guidelines.
- Skills in [.github/skills/](.github/skills/) for modular extensions.

## License

MIT License.
