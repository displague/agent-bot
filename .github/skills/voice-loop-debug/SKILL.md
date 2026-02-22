---
name: voice-loop-debug
description: Diagnose offline continuous voice loop issues in agent-bot, especially when inputs enqueue but no spoken or text response appears. Use for voice activity mismatches, silent listening states, and Esc debug-screen validation.
---

# Voice Loop Debug

Use this skill when the app shows voice loop running but user sees no response.

## Quick Checks

1. Confirm runtime status from UI commands:
   - `/voice-status`
   - `/voice-diagnose`
   - `/llm-status`
2. Capture key fields:
   - voice mode/activity (`listening`, `thinking`, `error`)
   - `unprocessed` count trend
   - llm `processing` flag and `last_error`
3. Verify Esc debug view is populated (not blank). If blank, inspect TUI debug render path first.

## Triage Flow

1. If voice loop is `running` and `listening` but no replies:
   - Submit typed input and observe whether `unprocessed` decreases.
   - If it does not decrease, inspect interaction processor + queue handoff.
2. If state flips to `processing utterance` and stalls:
   - Run `/llm-diagnose` and inspect timeout/error path.
   - Check model backend and current alias for overload/offload behavior.
3. If `voice loop activity=error`:
   - Surface the full stderr cause (avoid truncation in status line).
   - Reproduce once with deterministic text input before touching audio.

## File Touchpoints

- `voice_loop.py`
- `interaction_processor.py`
- `tui_renderer.py`
- `simple_renderer.py`

## Fix Patterns

- Add bounded timeouts for inference and shutdown waits.
- Ensure queued items are always drained or re-queued on failure.
- Keep diagnostics commands lightweight and non-blocking.
- Prefer explicit status messages over implicit spinner-only states.
