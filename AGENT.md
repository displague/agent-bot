# Agent Collaboration Guidelines for Agent-Bot

This document is the primary context handoff document for AI agents (Copilot, Gemini, etc.) working on agent-bot. Read this fully before making any changes. It describes the current working state, critical bugs fixed, ongoing research, and known failure modes.

## Project Summary

Agent-bot is an autonomous AI system with:
- **Voice backbone**: NVIDIA PersonaPlex (Moshi 7B) running offline, full-duplex, in-process on a local GPU
- **Reasoning backbone**: `google/gemma-3n-E2B-it` for multi-phase text/audio/vision reasoning
- **TUI**: curses-based terminal interface with port 9999 JSON debug socket
- **Branch**: `multiple_files` (active development)

## Current State (as of 2026-02-26)

All voice pipeline critical bugs have been fixed. The system can hear user speech and generate varied (non-deterministic) spoken responses. Inference is slow (~500ms/frame vs 80ms budget) due to full-precision 7B on maxed VRAM — this is a hardware/quantization research problem, not a code bug.

### Fixed Bugs (do not re-introduce)

| Bug | File | Fix | Commit |
|---|---|---|---|
| `top_k=1` greedy → identical output every call | `utils.py` | `top_k=250, top_k_text=25` | `aff5a32` |
| `other_mimi.decode()` used instead of `mimi.decode()` (Bug A) | `utils.py` | `mimi.decode(tokens[:,1:9])` in `step/infer_stream/infer` | `3079cee` |
| `other_mimi` state sync missing (Bug B) | `utils.py` | `_ = other_mimi.encode(chunk)` + `_ = other_mimi.decode(tokens[:,1:9])` every frame | `3079cee` |
| `_streaming_session` race: `None` between check and `.step()` | `voice_loop.py` | Capture `session = self._streaming_session` locally; guard `if session is None: continue` | `e73a56d` |
| Hard-interruption log flood | `voice_loop.py` | `if not self._playback_interrupt.is_set():` guard before `set()` + log | `e73a56d` |
| `/voice-diagnose` torch check syntax error on Windows | `simple_renderer.py`, `tui_renderer.py` | Rewrite `check_script` as multiline string (can't use `try: stmt; stmt` inline) | `aff5a32` |
| `_voice_prompt_cache` blocking os.walk on every call | `utils.py` | Module-level dict + search `voices/` first | earlier |
| `step_system_prompts` ~46s called every turn | `utils.py` | `_save_primed_state()` / `_restore_primed_state()` KV-cache snapshot to CPU | earlier |

### Mimi Routing (Critical — must stay correct)

The correct pattern for every inference frame in `step()`, `infer_stream()`, and `infer()`:

```python
# User audio → model input (both must encode every frame for state sync)
codes = mimi.encode(chunk)
_ = other_mimi.encode(chunk)        # discard — state sync only

# Model inference
tokens = lm_gen.step(codes)

# Decode output (use mimi NOT other_mimi for real audio output)
pcm = mimi.decode(tokens[:, 1:9])
_ = other_mimi.decode(tokens[:, 1:9])  # discard — state sync only
```

`tts_stream()` uses `mimi.decode()` only (no user audio input) — that was already correct.

### Sampling Parameters (LMGen in utils.py ~line 496)

```python
self.lm_gen = LMGen(
    self.lm,
    use_sampling=True,
    temp=0.8,          # temperature for audio tokens
    temp_text=0.7,     # temperature for text tokens
    top_k=250,         # MLX reference default — DO NOT set to 1 (greedy)
    top_k_text=25,     # MLX reference default — DO NOT set to 1 (greedy)
    audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),  # 6 frames
    ...
)
```

## Roles

- **Audio Processing Agent**: Manages `AudioMultiplexer` and `PersonaPlexManager`. Orchestrates full-duplex streaming, in-process model inference, and reflexive verbal fillers.
- **Sensory Agent**: Maintains the `RollingAudioBuffer` (auditory memory) and provides `inspect_audio_snippet` for multi-modal context analysis.
- **Reasoning Agent**: Oversees multi-phase processing (Notes, Planning, Execution, Digesting, Validating, Responding) using Gemma-3n. Ensures conversation history strictly alternates roles.
- **UI Agent**: Manages the TUI with enhanced navigation: Up/Down for input history, PgUp/PgDn for log scrolling, and Left/Right for cursor editing.
- **Logging Agent**: Ensures a unified log schema via `InteractionLogManager`, facilitating accurate summaries by the `EventCompressor`.

## Tools Available

- **Model Inference**: Gemma-3n (multi-modal transformer) and legacy llama.cpp backends. `PersonaPlexManager` keeps audio models warm in VRAM for rapid response.
- **Auditory Backbone**: `AudioMultiplexer` for broadcast mic capture and `RollingAudioBuffer` for 10s sensory memory.
- **TUI Controls**: Curses-based interface with `/wake`, `/sleep`, and real-time processing phase visibility.
- **Shutdown Watchdog**: 10-second hard-kill watchdog ensuring reliable process termination.
- **Debug Socket**: Port 9999 — send JSON lines to interact with a running agent. See Debug Commands below.

## Interaction Protocols

- **Communication**: Use role prefixes and reference the current processing phase (e.g., "[Reasoning Agent] Phase: Planning").
- **Handoff Logic**: Priority 1: Trigger reflexive filler. Priority 2: Multi-phase background reasoning. Priority 3: Final spoken response.
- **Context Management**: Explicitly alternate 'user' and 'assistant' roles in `llm_context` to comply with Gemma-3n constraints. Merge consecutive entries from the same role.
- **GPU Serialization**: Always acquire `_processing_lock` before making model calls (LLM or PersonaPlex) to prevent CUDA graph conflicts and VRAM spikes.
- **State Tracking**: Monitor `manual_wake` status and "Next event" time-tracking for autonomous operations.

## Guidelines

- **Code Quality**: Follow Python best practices; add docstrings and comments. Use async/await for concurrency.
- **Security**: Protect HF_TOKEN and avoid logging sensitive data.
- **VRAM**: Total is ~16.32 GB (LM: 15.59 GB + mimi: 0.73 GB). Card is maxed. Do NOT introduce extra model copies. `torch.no_grad()` is on all inference paths.
- **Thread Safety**: Use `loop.call_soon_threadsafe` when broadcasting from background capture threads to async queues.
- **Windows UTF-8**: Always set `PYTHONUTF8=1` (or `$env:PYTHONUTF8='1'`) before running on Windows. The Moshi tokenizer fails with charmap errors without it.
- **Testing**: Validate changes with unit tests (pytest). Use the global mocking strategy in `tests/conftest.py` to avoid heavy model loads. To test real logic that is globally mocked, use `@pytest.mark.skip_heavy_mock`. Always set a timeout (e.g., `--timeout=30`) when running tests.
- **Collaboration**: Propose changes via pull requests; review for integration impact.
- **Documentation**: Update this file and README.md as features evolve.

## PersonaPlex Specifics

- **Manager**: Use `PersonaPlexManager` for in-process inference. Avoid subprocess calls unless debugging.
- **Voices**: Default to `NATF2.pt` for natural conversational tone. `voices/` directory is gitignored (personal audio files).
- **Modes**:
  - Full-duplex streaming is the primary runtime path via `PersonaPlexStreamingSession`.
  - `tts_stream`: Teacher-forces text tokens; depformer samples audio autoregressively — fast speech without a conversational exchange.
  - `hear_stream`: Generates pyttsx3 TTS of "heard" text, feeds to `infer()`, yields model's conversational reply.
  - `infer`: Single-turn WAV-in/WAV-out inference (used by `hear_stream`).
- **Voice primer**: `voices/introduce-yourself.wav` — a 24 kHz WAV of "Please introduce yourself." ready to feed to `hear_stream` or `infer`.
- **KV-Cache lifecycle**:
  1. `load()` runs `step_system_prompts()` once (~46-89s), then `mimi.reset_streaming()`, then `_save_primed_state()` (CPU snapshot)
  2. Each call to `tts_stream`/`infer_stream`/`infer` starts with `_restore_primed_state()` (~1s)
  3. `PersonaPlexStreamingSession.start()` also uses `_restore_primed_state()`
  4. After TTS completes, `_streaming_session = None` is set so the next voice frame triggers a fresh session

## Running the Agent

```powershell
# Standard voice-only run:
$env:PYTHONUTF8='1'; uv run main.py --reset-logs --skip-deep-reasoning --no-sleep

# Watch logs in another terminal:
Get-Content logs\app.log -Wait -Tail 50
```

## Debug Commands (port 9999 + TUI)

Connect to port 9999 and send JSON lines:
```json
{"type": "command", "data": "/voice-say Hello world"}
```

Or use the TUI directly (type commands at the prompt).

| Command | Description |
|---|---|
| `/voice-say <text>` | Inject text as an `override_response`; spoken via `tts_stream` (bypass path). |
| `/voice-hear <text>` | Synthesise text as user speech via pyttsx3, feed to PersonaPlex conversationally, play agent reply. |
| `/voice-diagnose` | Print python path, torch/CUDA/moshi versions. Fixed in `aff5a32` — now works on Windows. |
| `/voice-start` / `/voice-stop` | Start/stop the offline continuous voice loop. |
| `/voice-status` | Show model load state, session status, VRAM usage. |
| `/voice-test-tone` | Play a test beep to verify audio output routing. |
| `/voice-devices` | List audio input/output devices. |
| `/voice-set-device out <id>` | Switch output audio device at runtime. |
| `/set-persona <text>` | Update `PERSONAPLEX_TEXT_PROMPT` at runtime. |
| `/logic-reload` | Hot-swap `functional_agent.py`, `interaction_processor.py`, `utils.py` without model reload. |
| `/llm-status` | Show LLM backend state. |
| `/llm-diagnose` | Verbose LLM diagnostics. |
| `/wake` / `/sleep` | Toggle autonomous sleep cycle. |
| `/quit` / `/force-quit` | Graceful or forced shutdown. |

## Project Skills

Read these before debugging specific issues:

- **voice-loop-debug** (`.github/skills/voice-loop-debug/SKILL.md`): Triage voice loop failures, VAD, streaming session issues, mimi routing, greedy decoding, KV-cache. **Read first for any voice problem.**
- **model-load-triage** (`.github/skills/model-load-triage/SKILL.md`): `step_system_prompts` latency, KV-cache save/restore, `streaming_forever` AssertionError, `_voice_prompt_cache` blocking walk.
- **vram-multi-modal-optimize** (`.github/skills/vram-multi-modal-optimize/SKILL.md`): VRAM limits, quantization ineffectiveness for Moshi, eager mode, graph capture stability.
- **env-cuda-alignment** (`.github/skills/env-cuda-alignment/SKILL.md`): Verify bfloat16 compatibility, `moshi` package visibility, single `.venv` usage.
- **multi-modal-diagnose** (`.github/skills/multi-modal-diagnose/SKILL.md`): `AutoProcessor` state, audio-tool feedback loops for Gemma-3n.
- **shutdown-recovery** (`.github/skills/shutdown-recovery/SKILL.md`): 10s watchdog, second Ctrl+C force kill, `/force-quit`.
- **debug-link** (`.github/skills/debug-link/SKILL.md`): How to connect to port 9999 hot socket.

## Open Research (GH Issues)

- **#8**: Post-load PyTorch quantization yields only ~0.21 GB VRAM savings for Moshi (large layers must be skipped). bitsandbytes or GGUF needed for real reduction.
- **#9**: MLX reference project downloads pre-quantized `model.q4.safetensors` from `nvidia/personaplex-7b-v1` — investigate if PyTorch pipeline can use these.
- **#5**: Incoherent streaming tokens (likely improved by `top_k=250` fix — needs verification).
- **#7**: Voice-hear response quality (likely improved by `top_k=250` fix — needs verification).

## Completed Phases

### Phase 6: Audio Multiplexer
**DONE** — `AudioMultiplexer` + `RollingAudioBuffer` for thread-safe multi-subscriber audio capture.

### Phase 7: True Full-Duplex
**DONE** — `PersonaPlexStreamingSession` processes `LMGen.step` in real-time 80ms frames.

### Phase 8: Vision Integration
**DONE** — `capture_screen` + `inspect_current_screen` tool + multi-modal visual analysis in `llm_call`.

### Phase 11: Performance & Flow Overhaul
**DONE** — Warm Generator, Streaming Playback, Port 9999 Hub, `/logic-reload` hot-swap, moshi monkeypatches.

### Phase 12: Memory & Resource Orchestration
**PARTIAL** — `_save_primed_state` / `_restore_primed_state` eliminates `step_system_prompts` repeat. Real VRAM reduction (Phase 12 goal) blocked on quantization research (#8/#9).

Agents should operate autonomously while coordinating for seamless development.

