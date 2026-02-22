---
name: shutdown-recovery
description: Recover from stuck shutdowns in agent-bot and implement robust termination semantics, including second-Ctrl+C force kill, /force-quit, and process-tree cleanup.
---

# Shutdown Recovery

Use this skill when app hangs on "Starting shutdown sequence" or Ctrl+C does not exit.

## Recovery Procedure

1. Attempt graceful shutdown once.
2. Observe the 10-second hard-kill watchdog:
   - If cleanup exceeds 10s, "Shutdown watchdog triggered" should appear.
   - Confirm the process tree is killed via `force_exit_now`.
3. If capture thread hangs:
   - Check `AudioMultiplexer.stop()` and `join()` logic.

## Implementation Rules

- Graceful path must be bounded by individual component timeouts (e.g., 2s for scheduler).
- Watchdog must be independent (daemon thread) to ensure it triggers regardless of main-loop blocking.
- Cleanup should include stopping `RollingAudioBuffer` and `AudioMultiplexer`.

## Validation

1. Start voice loop and ensure capture is active.
2. Press Ctrl+C once.
3. Confirm "Starting shutdown sequence" log appears.
4. Verify process exits within 10s (graceful) or exactly at 10s (watchdog).

## File Touchpoints

- `main.py`
- `process_utils.py`
- `voice_loop.py`
- `tui_renderer.py`
- `simple_renderer.py`
