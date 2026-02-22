# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent that can respond to queries, compress historical data, and schedule future actions.

## Features

- **Bidirectional Audio Interaction**: Uses NVIDIA PersonaPlex for real-time, full-duplex speech-to-speech conversations with persona control and voice conditioning.
- **Multi-Phase Reasoning**: Breaks down tasks into planning, execution, digestion, validation, and response phases.
- **Model Switching**: Runtime `/model` command to inspect and switch between configured model backends (default is `gpt-oss`).
- **Autonomous Operation**: Generates periodic "thoughts" during active hours, compresses interaction logs hourly, and schedules events like reminders or training.
- **Logging and Indexing**: Maintains hard logs (JSONL), compressed summaries, and a keyword-based index for context retrieval.
- **TUI Interface**: Real-time interface using curses, showing status, thoughts, interaction history, and debug info. Supports text input or voice (via PersonaPlex integration).
- **Event Management**: Handles deferred topics, lookups, RAG completions, and training events.
- **State Management**: Tracks unprocessed interactions, ongoing thoughts, sleep status, and next events.

## Architecture

- `main.py`: Entry point; initializes and runs all components asynchronously using curses for TUI.
- `runtime_manager.py`: Manages shared executors and tracked async tasks for graceful shutdown.
- `tui_renderer.py`: Manages the curses-based TUI; handles rendering, input (text/voice), scrolling, and screen switching.
- `interaction_processor.py`: Dequeues interactions, processes via `FunctionalAgent`, logs to files, indexes entries, and updates state.
- `functional_agent.py`: Orchestrates multi-phase processing using `LlamaModelManager`.
- `llama_model_manager.py`: Manages model backends (`llama_cpp` and `transformers`), HF-cache model resolution, and runtime switching.
- `thought_generator.py`: Generates autonomous thoughts during active hours using `FunctionalAgent`.
- `event_scheduler.py`: Async queue for events; handles scheduling and execution.
- `event_compressor.py`: Hourly compression of logs using Llama to summarize interactions; saves compressed logs and triggers RAG events.
- `interaction_log_manager.py`: In-memory log with async locking; provides paginated display data.
- `index_manager.py`: Builds/saves a JSON index of interactions by keywords; supports context searches.
- `utils.py`: Utility functions; includes speech-to-speech functions via PersonaPlex.
- `voice_loop.py`: Continuous offline voice loop with VAD segmentation, state transitions, and interruption handling.
- `config.py`: Constants for paths (logs, index), sleep times, workers, and model catalog/aliases.
- `logging_setup.py`: Configures logging to files and stderr redirection.

## Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- Opus audio codec: Install `libopus-dev` (Ubuntu/Debian) or equivalent.
- Hugging Face account with access to [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)

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

3. Install Python dependencies into `agent-bot/.venv`:
   ```bash
   uv pip install --python .venv/Scripts/python.exe -r requirements.txt
   ```

4. Install PersonaPlex:
   ```bash
   git clone https://github.com/NVIDIA/personaplex.git
   # dependencies already include editable ./personaplex/moshi via requirements.txt
   # For Blackwell GPUs, install CUDA-enabled torch in .venv as needed.
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

Quick first-run validation:
1. Start with simple UI to avoid terminal/curses issues:
   ```bash
   # PowerShell
   $env:AGENTBOT_UI_MODE='simple'; $env:AGENTBOT_DEV_MODE='1'; python main.py
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
