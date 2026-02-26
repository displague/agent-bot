# Agent-Bot

An autonomous AI agent system that interacts via text or voice, processes interactions through multi-phase reasoning, and provides a terminal-based user interface (TUI). It continuously operates, simulating an autonomous agent that can respond to queries, compress historical data, and schedule future actions.

## Features

- **True Full-Duplex Audio**: Real-time speech-to-speech interaction via NVIDIA PersonaPlex (Moshi 7B). The agent listens and responds simultaneously via continuous `PersonaPlexStreamingSession`.
- **Warm Generator Architecture**: Maintains a warm `LMGen` instance in VRAM, with KV-cache saved to CPU after `step_system_prompts`. Each turn restores from the primed state (~1s) instead of re-running system prompts (~46s).
- **Auditory & Visual Memory**: 10-second rolling audio buffer and native screen capture for rich contextual reasoning.
- **Multi-Modal Reasoning**: Powered by `google/gemma-3n-E2B-it`, supporting native text, audio, and vision analysis with 4-bit quantization.
- **Real-Time Telemetry**: Detailed VRAM and duration tracking for all loading stages and inference turns.
- **Experimentation Hub (Hot Socket)**: Port 9999 JSON socket for live command injection, state monitoring, and remote interaction.
- **Deep Logic Hot-Reload**: `/logic-reload` command allows for runtime swapping of `functional_agent.py`, `interaction_processor.py`, and `utils.py` without losing model weights.
- **Advanced Audio Diagnostics**: Live device switching, hardware verification tones, and explicit device routing logs.
- **Robust Shutdown**: Graceful shutdown sequence with a 10-second hard-kill watchdog.

## Architecture

- `main.py`: Entry point; manages lifecycle, CLI flags, and the JSON `DebugServer` on port 9999.
- `utils.py`: Core infrastructure — `PersonaPlexManager` (warm weights, KV-cache save/restore, mimi routing), `PersonaPlexStreamingSession` (per-turn full-duplex), `AudioMultiplexer`, `RollingAudioBuffer`. All inference paths wrapped in `torch.no_grad()`.
- `voice_loop.py`: Orchestrates VAD listener, streaming session lifecycle, playback loop, and hard-interruption logic.
- `interaction_processor.py`: Manages the cognitive handoff and supports `SKIP_DEEP_REASONING` for lightweight PersonaPlex testing.
- `tui_renderer.py` / `simple_renderer.py`: Renderers that manage the full `InteractionProcessor` lifecycle to support logic hot-swapping.
- `config.py`: Central configuration for audio thresholds, optimization strategies, and persona prompts. Key flags: `PERSONAPLEX_QUANTIZE`, `NO_SLEEP`.
- `quantize.py`: Per-channel int8/4bit post-load quantization (NOTE: ~0.21 GB VRAM reduction only — large Moshi fused-op layers must be skipped; see GH #8/#9 for real quantization path via pre-quantized HF files).

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 5080 recommended, 16GB+ VRAM)
- CUDA 12.1+ (Tested with CUDA 13.0)
- `uv` for environment management
- `triton-windows` (included in lockfile for Windows stability)

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

4. Install PersonaPlex (the moshi Python package is the upstream dependency):
   ```bash
   # Already included in requirements.txt — no separate step needed.
   # Model weights (~16 GB) are downloaded automatically from nvidia/personaplex-7b-v1 on first run.
   ```

### Running

```powershell
# Standard run (recommended for voice testing):
$env:PYTHONUTF8='1'; uv run main.py --reset-logs --skip-deep-reasoning --no-sleep

# With 4-bit quantization (saves ~0.21 GB VRAM — minimal; prefer no quantize unless testing):
$env:PYTHONUTF8='1'; uv run main.py --reset-logs --skip-deep-reasoning --no-sleep --quantize 4bit
```

CLI flags:
- `--reset-logs`: Clear interaction and debug logs on startup.
- `--dev`: Disable autonomous thoughts and hourly compression.
- `--skip-deep-reasoning`: Bypass Gemma-3n load; focus entirely on PersonaPlex voice.
- `--no-sleep`: Disable auto-sleep mode (keeps voice loop active for debugging).
- `--quantize {8bit,4bit}`: Apply per-channel post-load quantization (see VRAM note above).

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

## Configuration

- `config.py`: Adjust paths, thresholds, and persona prompts. Key vars: `PERSONAPLEX_QUANTIZE`, `NO_SLEEP`, `PERSONAPLEX_TEXT_PROMPT`.
- `AGENTBOT_UI_MODE`: `auto` (default), `simple`, or `curses`.
- `AGENTBOT_DEV_MODE`: `1` disables autonomous background tasks.
- `AGENTBOT_LOG_LEVEL`: default `INFO`; set to `DEBUG` for verbose diagnostics.
- `PYTHONUTF8=1`: **Required on Windows** to avoid charmap decode errors with Moshi tokenizer.

## Known Issues & Ongoing Research

| Issue | Status | Notes |
|---|---|---|
| Inference speed ~500ms/frame (real-time budget: 80ms) | Open | Bottleneck: full-precision 7B on maxed VRAM. Needs real quantization. |
| `--quantize` yields only ~0.21 GB VRAM savings | Open | GH #8/#9: large Moshi layers must be skipped; MLX project downloads pre-quantized HF files instead |
| `/voice-diagnose` torch check | Fixed `aff5a32` | Was invalid inline Python syntax on Windows; now multiline script |
| Greedy decoding → always same output | Fixed `aff5a32` | `top_k=1` + primed KV-cache = deterministic. Now `top_k=250, top_k_text=25` |
| Mimi routing (Bug A+B) | Fixed `3079cee` | `other_mimi.decode` → `mimi.decode`; `other_mimi.encode` discard for state sync |
| `_streaming_session` race crash | Fixed `e73a56d` | Capture local ref before `run_in_executor` |
| Hard-interruption log flood | Fixed `e73a56d` | Debounced with `if not self._playback_interrupt.is_set()` guard |

## Development

- See [AGENT.md](AGENT.md) for collaboration guidelines and current architecture decisions.
- Skills in [.github/skills/](.github/skills/) for modular diagnostic playbooks.
- GH Issues track open research: #8 (quantization), #9 (pre-quantized HF files), #5/#7 (voice response quality).
- Branch: `multiple_files` (active development)

## License

MIT License.

