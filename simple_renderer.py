import asyncio
import logging
import os
import subprocess
import textwrap
from datetime import datetime

from config import VOICE_AUTO_START_ON_LAUNCH, VOICE_CAPTURE_KEY_LABEL
from config import LLM_DIAG_TIMEOUT_SECONDS
from process_utils import force_exit_now
from smoke_test_runner import (
    run_deterministic_smoke,
    run_model_smoke,
    summarize_smoke_result,
)
from utils import _resolve_personaplex_python
from voice_loop import VoiceLoop

logger = logging.getLogger("autonomous_system.simple_renderer")


class SimpleRenderer:
    """Cross-platform fallback renderer when curses is unavailable."""

    def __init__(
        self,
        state,
        interaction_queue,
        interaction_log_manager,
        functional_agent=None,
        voice_loop=None,
    ):
        self.state = state
        self.interaction_queue = interaction_queue
        self.interaction_log_manager = interaction_log_manager
        self.functional_agent = functional_agent
        self._stop = False
        self._last_log_offset = 0
        self._log_pump_task = None
        self._voice_loop = voice_loop or VoiceLoop(state, interaction_log_manager)

    async def start(self):
        print("Simple mode active (no curses).")
        print("Type /help for commands. Type /quit to exit.")
        if self.state.get("voice_mode") != "offline-continuous":
            self.state["voice_mode"] = "offline-disabled"
        self._log_pump_task = asyncio.create_task(self._pump_logs())
        if VOICE_AUTO_START_ON_LAUNCH:
            try:
                await self._voice_loop.start()
                print("Voice loop auto-started.")
            except Exception as e:
                print(f"Voice auto-start failed: {str(e)[:120]}")
        try:
            while not self._stop:
                try:
                    line = await asyncio.to_thread(input, "Input> ")
                    if not line:
                        continue
                    if line.startswith("/"):
                        await self.handle_command(line)
                        continue
                    if line.strip().lower() in {"/quit", "/exit"}:
                        self._stop = True
                        break
                    self.interaction_queue.put_nowait(
                        {
                            "input": line,
                            "private_notes": "",
                            "audio_waveform": None,
                        }
                    )
                    self.state["unprocessed_interactions"] += 1
                    await self.interaction_log_manager.append(f"Input: {line}")
                    print("Queued input. Waiting for response...")
                except (EOFError, KeyboardInterrupt):
                    self._stop = True
                except Exception as e:
                    logger.error("Simple renderer error: %s", e)
                    await asyncio.sleep(0.1)
        finally:
            await self._voice_loop.stop()
            if self._log_pump_task is not None:
                self._log_pump_task.cancel()
                await asyncio.gather(self._log_pump_task, return_exceptions=True)

    async def _pump_logs(self):
        while not self._stop:
            try:
                entries, next_offset = await self.interaction_log_manager.get_entries_since(
                    self._last_log_offset
                )
                self._last_log_offset = next_offset
                for entry in entries:
                    print(f"[log] {entry}")
            except Exception as e:
                logger.error("Simple renderer log pump error: %s", e)
            await asyncio.sleep(0.2)

    async def handle_command(self, command: str):
        raw = command.strip()
        cmd = raw.lower()
        parts = raw.split()
        model_manager = (
            getattr(self.functional_agent, "llama_manager", None)
            if self.functional_agent is not None
            else None
        )
        if cmd in {"/quit", "/exit"}:
            self._stop = True
            return
        if cmd == "/force-quit":
            force_exit_now(130)
        if cmd == "/help":
            print(
                "Commands: /help, /model, /llm-status, /llm-diagnose, /smoke, /smoke-model, /smoke-all, "
                "/voice-start, /voice-stop, /voice-status, /voice-diagnose, /wake, /sleep, /quit, /force-quit"
            )
            print("Ctrl+D (EOF) also exits in simple mode.")
            print(f"Voice hotkey is available in curses mode via {VOICE_CAPTURE_KEY_LABEL}.")
            return
        if cmd == "/model" or cmd.startswith("/model "):
            if model_manager is None:
                print("Model manager unavailable.")
                return
            if len(parts) == 1:
                info = model_manager.get_model_info()
                aliases = ", ".join(model_manager.list_models().keys())
                print(
                    f"Model active: {info.get('alias')} ({info.get('backend_active')}); "
                    f"available: {aliases}. Use /model use <alias>."
                )
                return
            if len(parts) == 2 and parts[1].lower() in {"list", "ls"}:
                catalog = model_manager.list_models()
                details = ", ".join(
                    f"{name}({spec.get('backend')})" for name, spec in catalog.items()
                )
                print(f"Available models: {details}")
                return
            if len(parts) == 3 and parts[1].lower() == "use":
                alias = parts[2]
                print(f"Loading model '{alias}'...")
                try:
                    info = await asyncio.to_thread(model_manager.load_model, alias)
                    print(
                        f"Model switched to {info.get('alias')} "
                        f"({info.get('backend_active')})."
                    )
                except Exception as e:
                    print(f"Model switch failed: {str(e)[:160]}")
                return
            print("Usage: /model | /model list | /model use <alias>")
            return
        if cmd == "/llm-status":
            info = model_manager.get_model_info() if model_manager is not None else {}
            llm_busy = model_manager.is_busy() if model_manager is not None else False
            elapsed = ""
            started_at = self.state.get("last_processing_started_at")
            if self.state.get("is_processing") and started_at:
                try:
                    dt = datetime.fromisoformat(started_at)
                    elapsed = f"{(datetime.now() - dt).total_seconds():.1f}s"
                except Exception:
                    elapsed = "unknown"
            msg = (
                f"LLM status: alias={info.get('alias')} backend={info.get('backend_active')} "
                f"busy={llm_busy} "
                f"processing={self.state.get('is_processing')} "
                f"elapsed={elapsed} "
                f"unprocessed={self.state.get('unprocessed_interactions')} "
                f"last_status={self.state.get('last_processing_status', '')} "
                f"last_error={self.state.get('last_processing_error', '')}"
            )
            print("\n".join(textwrap.wrap(msg, 120)))
            return
        if cmd == "/llm-diagnose":
            if model_manager is None:
                print("LLM diagnose: model manager unavailable.")
                return
            if model_manager.is_busy():
                msg = "LLM diagnose: skipped (LLM busy with active request)."
                print(msg)
                await self.interaction_log_manager.append(msg)
                return
            try:
                lines = await asyncio.wait_for(
                    asyncio.to_thread(self._diagnose_llm_runtime, model_manager),
                    timeout=LLM_DIAG_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                lines = [f"LLM diagnose timeout after {LLM_DIAG_TIMEOUT_SECONDS}s"]
            for line in lines:
                print(line)
                await self.interaction_log_manager.append(line)
            return
        if cmd in {"/smoke", "/smoke-all"}:
            deterministic_result = await run_deterministic_smoke()
            deterministic_summary = summarize_smoke_result(deterministic_result)
            print(deterministic_summary)
            await self.interaction_log_manager.append(deterministic_summary)
        if cmd in {"/smoke-model", "/smoke-all"}:
            if self.functional_agent is None:
                msg = "SMOKE model: FAIL functional agent unavailable"
                print(msg)
                await self.interaction_log_manager.append(msg)
                return
            print("Running model smoke test...")
            model_result = await run_model_smoke(self.functional_agent)
            model_summary = summarize_smoke_result(model_result)
            print(model_summary)
            await self.interaction_log_manager.append(model_summary)
            return
        if cmd == "/voice-start":
            if self._voice_loop.is_running:
                print("Voice loop already running.")
                return
            await self._voice_loop.start()
            print("Voice loop started.")
            return
        if cmd == "/voice-stop":
            await self._voice_loop.stop()
            print("Voice loop stopped.")
            return
        if cmd == "/voice-status":
            print(
                f"Voice loop: {'running' if self._voice_loop.is_running else 'stopped'} "
                f"mode={self.state.get('voice_mode')} "
                f"activity={self.state.get('voice_activity_state')}"
            )
            return
        if cmd == "/voice-diagnose":
            for line in self._diagnose_voice_runtime():
                print(line)
                await self.interaction_log_manager.append(line)
            return
        if cmd == "/wake":
            self.state["manual_wake"] = True
            self.state["is_sleeping"] = False
            msg = "Agent manually woken from sleep."
            print(msg)
            await self.interaction_log_manager.append(msg)
            return
        if cmd == "/sleep":
            self.state["manual_wake"] = False
            msg = "Agent returned to autonomous sleep cycle."
            print(msg)
            await self.interaction_log_manager.append(msg)
            return
        if cmd not in {"/model", "/llm-status", "/llm-diagnose", "/smoke", "/smoke-all", "/force-quit"}:
            print(f"Unknown command: {command}. Try /help")

    def _diagnose_voice_runtime(self):
        lines = []
        py = _resolve_personaplex_python()
        lines.append(f"Voice diagnose: personaplex_python={py}")
        lines.append(f"Voice diagnose: python_exists={os.path.exists(py)}")
        command = [
            py,
            "-c",
            (
                "import torch; import moshi; "
                "print(f'torch={torch.__version__} moshi_available=True cuda_available={torch.cuda.is_available()} "
                "cuda_version={getattr(torch.version, \"cuda\", None)}')"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                lines.append(f"Voice diagnose: {result.stdout.strip()}")
            else:
                lines.append(f"Voice diagnose: torch check failed rc={result.returncode}")
                if result.stderr.strip():
                    lines.append(f"Voice diagnose stderr: {result.stderr.strip()[:120]}")
        except Exception as e:
            lines.append(f"Voice diagnose: failed to run torch check: {str(e)[:120]}")
        return lines

    def _diagnose_llm_runtime(self, model_manager):
        lines = []
        info = model_manager.get_model_info()
        lines.append(
            f"LLM diagnose: alias={info.get('alias')} backend={info.get('backend_active')}"
        )
        test_prompt = "Reply with exactly: OK"
        try:
            result = model_manager.llm_call(test_prompt, 8)
            lines.append(f"LLM diagnose result: {str(result).strip()[:160]}")
        except Exception as e:
            lines.append(f"LLM diagnose error: {str(e)[:200]}")
        return lines
