---
name: shutdown-recovery
description: Recover from stuck shutdowns in agent-bot and implement robust termination semantics, including second-Ctrl+C force kill, /force-quit, and process-tree cleanup.
---

# Shutdown Recovery

Use this skill when app hangs on "Interrupted/Shutting down" or Ctrl+C does not exit.

## Recovery Procedure

1. Attempt graceful shutdown once.
2. If still running after a short timeout, trigger hard stop:
   - second `Ctrl+C` force path
   - `/force-quit` command path
3. Ensure the full process tree is terminated, not only parent PID.

## Implementation Rules

- Graceful path must be bounded by timeout.
- Force path must be immediate and explicit in UI output.
- Any blocked worker/thread should not prevent process exit in force mode.
- Keep cleanup idempotent (safe to call multiple times).

## Validation

1. Start voice loop.
2. Trigger a synthetic long-running inference.
3. Press Ctrl+C once (graceful), then again (force).
4. Confirm process exits and prompt returns promptly.

## File Touchpoints

- `main.py`
- `process_utils.py`
- `voice_loop.py`
- `tui_renderer.py`
- `simple_renderer.py`
