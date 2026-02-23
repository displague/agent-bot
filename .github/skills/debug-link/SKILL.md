---
name: debug-link
description: Provides instructions for connecting to the Agent-Bot 'Hot Socket' (DebugServer) on Port 9999 for live command injection and monitoring.
---

# Debug Link: Port 9999 Hot Socket

Use this skill when you need to interact with or monitor a running Agent-Bot instance without restarting it.

## Quick Connection (via Python)

Run this one-liner to send a command to the running app:

```bash
python -c "import socket, json; s=socket.create_connection(('127.0.0.1', 9999)); s.sendall((json.dumps({'type': 'command', 'data': '/voice-status'}) + '
').encode()); print(s.recv(1024).decode())"
```

## Available Message Types

### 1. Command Injection
```json
{"type": "command", "data": "/voice-say Hello from the socket"}
```
- Injects any slash command into the app's primary command handler.
- Supports `/logic-reload`, `/voice-test-tone`, `/set-persona`, etc.

### 2. State Inspection
```json
{"type": "state"}
```
- Returns a JSON object of the current system `state`.

## Core Developer Commands

- `/logic-reload`: Hot-swaps `functional_agent.py` and `interaction_processor.py`.
- `/voice-say <text>`: Direct TTS injection (bypasses mic/VAD).
- `/set-persona <text>`: Live update of the `PERSONAPLEX_TEXT_PROMPT`.
- `/voice-test-tone`: Verifies audio output hardware.
