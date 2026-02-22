# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent that can respond to queries, compress historical data, and schedule future actions.

## Features

- **True Full-Duplex Audio**: Real-time speech-to-speech interaction with zero-latency streaming. The agent can interject or respond instantly (<1s) while concurrently processing environmental audio.
- **Warm Generator Architecture**: Maintains a warm `LMGen` instance in VRAM to eliminate the multi-minute model re-instantiation penalty between turns.
- **Auditory & Visual Memory**: 10-second rolling audio buffer and native screen capture for rich contextual reasoning.
- **Multi-Modal Reasoning**: Powered by `google/gemma-3n-E2B-it`, supporting native text, audio, and vision analysis with 4-bit quantization.
- **Real-Time Telemetry**: Detailed VRAM and duration tracking for all loading stages and inference turns.
- **Experimentation Hub (Hot Socket)**: Port 9999 JSON socket for live command injection, state monitoring, and remote interaction.
- **Deep Logic Hot-Reload**: `/logic-reload` command allows for runtime swapping of `functional_agent.py`, `interaction_processor.py`, and `utils.py` without losing model weights.
- **Advanced Audio Diagnostics**: Live device switching, hardware verification tones, and explicit device routing logs.
- **Robust Shutdown**: Graceful shutdown sequence with a 10-second hard-kill watchdog.

## Architecture

- `main.py`: Entry point; manages the high-level lifecycle and the JSON `DebugServer`.
- `utils.py`: Core infrastructure including `AudioMultiplexer`, `PersonaPlexManager` (warm weights), and the Moshi streaming state bug-fixes.
- `voice_loop.py`: Orchestrates the listener, processor, and the new asynchronous `_playback_loop` for chunked audio streaming.
- `interaction_processor.py`: Manages the cognitive handoff and supports `SKIP_DEEP_REASONING` for lightweight PersonaPlex testing.
- `tui_renderer.py` / `simple_renderer.py`: Renderers that now manage the full `InteractionProcessor` lifecycle to support logic hot-swapping.
- `config.py`: Central configuration for audio thresholds, optimization strategies, and persona prompts.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- CUDA 12.1+ (Tested with CUDA 13.0)
- `triton-windows` (included in lockfile for Windows stability)

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

### Running

```bash
python main.py
```

Optional flags:
- `--reset-logs`: Clear interaction and debug logs on startup. (Preserves model weights/shards).
- `--dev`: Disable autonomous thoughts and hourly compression.
- `--skip-deep-reasoning`: Bypass Gemma-3n load to focus entirely on PersonaPlex.

Quick first-run validation:
1. Start with simple UI and reset logs:
   ```bash
   # PowerShell
   $env:AGENTBOT_UI_MODE='simple'; python main.py --reset-logs --dev --skip-deep-reasoning
   ```
2. At the prompt, run:
   - `/help`
   - `/voice-test-tone` (Verify audio output)
   - `/voice-status` (Check model health)
   - `/logic-reload` (Test hot-swapping)
   - `/quit` to exit cleanly

## Configuration

- `config.py`: Adjust paths, thresholds, and persona prompts.
- `AGENTBOT_UI_MODE`: `auto` (default), `simple`, or `curses`.
- `AGENTBOT_DEV_MODE`: `1` disables autonomous background tasks.
- `AGENTBOT_LOG_LEVEL`: default `INFO`; set to `DEBUG` for verbose diagnostics.

## Development

- See [AGENT.md](AGENT.md) for collaboration guidelines.
- Skills in [.github/skills/](.github/skills/) for modular extensions.

## License

MIT License.
