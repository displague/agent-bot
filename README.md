# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent that can respond to queries, compress historical data, and schedule future actions.

## Features

- **True Zero-Latency Voice**: The agent starts speaking fillers and conversational responses as soon as the first chunk is generated (<1s), bypassing intermediate WAV files.
- **Enhanced Real-Time Telemetry**: Live VRAM usage, inference latency (ms/frame), recent text tokens, and loading stages visible directly on the TUI debug screen.
- **Robust Model Loading**: CPU-first weight loading for quantized models to prevent VRAM OOM on 16GB GPUs. Supports pre-quantized `bitsandbytes` weights from community repositories.
- **Auditory & Visual Memory**: 10-second rolling audio buffer and native screen capture for rich contextual reasoning.
- **Multi-Modal Reasoning**: Powered by `google/gemma-3n-E2B-it`, supporting native text, audio, and vision analysis.
- **Experimentation Hub (Hot Socket)**: Port 9999 JSON socket for live command injection and remote state monitoring.
- **Deep Logic Hot-Reload**: `/logic-reload` command allows for runtime swapping of code without losing model weights.
- **Advanced Audio Diagnostics**: Live device switching, hardware verification tones, and explicit device routing logs.
- **Robust Shutdown**: Graceful shutdown sequence with a 10-second hard-kill watchdog.

## Architecture

- `main.py`: Entry point; refactored to start TUI early for visibility during model loading. Manages the JSON `DebugServer` on port 9999.
- `utils.py`: Core infrastructure — `PersonaPlexManager` (pre-quantized weight loading, CPU-first safety, monkeypatching), `PersonaPlexStreamingSession` (latency tracking), `AudioMultiplexer`.
- `voice_loop.py`: Orchestrates VAD listener and `say_stream()` chunked playback with precise gapless timing.
- `interaction_processor.py`: Manages the cognitive handoff using new zero-latency streaming paths.
- `quantize.py`: Integrates `bitsandbytes` NF4 quantization with bit-depth aware size estimation.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- CUDA 12.1+ (Tested with CUDA 13.0)
- `uv` for environment management
- `bitsandbytes` for real 4-bit quantization

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd agent-bot
   ```

2. Create and activate the project virtualenv:
   ```bash
   uv venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies from lockfile:
   ```bash
   uv pip install -r requirements.txt
   ```

### Running

```powershell
# Standard run (recommended for voice testing):
$env:PYTHONUTF8='1'; uv run main.py --reset-logs --skip-deep-reasoning --no-sleep

# With 4-bit quantization (Definitively fits in 16GB VRAM via bitsandbytes):
$env:PYTHONUTF8='1'; uv run main.py --reset-logs --skip-deep-reasoning --no-sleep --quantize 4bit
```

CLI flags:
- `--reset-logs`: Clear interaction and debug logs on startup.
- `--dev`: Disable autonomous thoughts and hourly compression.
- `--skip-deep-reasoning`: Bypass Gemma-3n load; focus entirely on PersonaPlex voice.
- `--no-sleep`: Disable auto-sleep mode (keeps voice loop active for debugging).
- `--quantize {8bit,4bit}`: Apply real weights quantization (bitsandbytes).

Quick first-run validation:
1. Start with simple UI and reset logs:
   ```powershell
   $env:AGENTBOT_UI_MODE='simple'; $env:PYTHONUTF8='1'; uv run main.py --reset-logs --dev --skip-deep-reasoning --no-sleep
   ```
2. At the prompt, run:
   - `/help`
   - `/voice-test-tone` (Verify audio output)
   - `/voice-status` (Check model health)
   - `/voice-diagnose` (Print torch/CUDA/moshi versions)
   - `/voice-start` (Start continuous voice loop)
   - `/voice-say Hello` (Inject TTS bypass)
   - `/voice-hear Hello` (Full conversational turn)
   - `/logic-reload` (Test hot-swapping)
   - `/quit` to exit cleanly

## Known Issues & Ongoing Research

| Issue | Status | Notes |
|---|---|---|
| Linguistic Drift during slowness | Research | GH #5: Inference speed ~500ms/frame (Budget: 80ms) causes codec drift. |
| Weights Load OOM | Fixed `5f6081c` | Resolved via CPU-first loading and GPU isolation during weight transfer. |
| Incoherent response | Mitigated | Improved via greedy text decoding (`temp_text=0.0`). |
| TUI Input Lag | Fixed `b31b96c` | Reduced refresh interval from 0.2s to 0.05s (20Hz). |
| repeated startup greeting | Fixed `b31b96c` | Explicit `lm_gen.reset_streaming()` during primed state restoration. |
| AttributeError in multi_linear | Fixed `2b3bd38` | Implemented `.weight` property in quantized layers for fused-op compatibility. |

## Development

- See [AGENT.md](AGENT.md) for collaboration guidelines and current architecture decisions.
- Skills in [.github/skills/](.github/skills/) for modular diagnostic playbooks.
- GH Issues track open research: #8 (quantization), #9 (pre-quantized HF files), #5/#7 (voice response quality).
- Branch: `multiple_files` (active development)

## License

MIT License.
