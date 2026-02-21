import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

from config import SMOKE_LOG_PATH, SMOKE_MODEL_TIMEOUT_SECONDS

SMOKE_REQUEST = "Hello, repeat after me, testing, testing, 1, 2, 3."
SMOKE_EXPECTED_RESPONSE = "Testing, testing, 1, 2, 3."


def _now_iso() -> str:
    return datetime.now().isoformat()


def _model_response_passes(response: str) -> bool:
    normalized = (response or "").lower()
    return "testing" in normalized and "1, 2, 3" in normalized


def summarize_smoke_result(result: Dict[str, Any]) -> str:
    status = "PASS" if result.get("passed") else "FAIL"
    mode = result.get("mode", "unknown")
    duration_ms = int(result.get("duration_ms", 0))
    detail = result.get("error") or result.get("response", "")
    detail = str(detail).strip().replace("\n", " ")
    if len(detail) > 120:
        detail = detail[:117] + "..."
    return f"SMOKE {mode}: {status} ({duration_ms}ms) {detail}"


def _append_smoke_record_sync(result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(SMOKE_LOG_PATH), exist_ok=True)
    with open(SMOKE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def _append_smoke_record(result: Dict[str, Any]) -> None:
    await asyncio.to_thread(_append_smoke_record_sync, result)


async def run_deterministic_smoke() -> Dict[str, Any]:
    start = time.perf_counter()
    response = SMOKE_EXPECTED_RESPONSE
    duration_ms = (time.perf_counter() - start) * 1000
    result = {
        "timestamp": _now_iso(),
        "mode": "deterministic",
        "request": SMOKE_REQUEST,
        "expected": SMOKE_EXPECTED_RESPONSE,
        "response": response,
        "passed": response == SMOKE_EXPECTED_RESPONSE,
        "duration_ms": duration_ms,
        "error": None,
    }
    await _append_smoke_record(result)
    return result


async def run_model_smoke(functional_agent) -> Dict[str, Any]:
    start = time.perf_counter()
    response = ""
    error = None
    try:
        llama_manager = getattr(functional_agent, "llama_manager", None)
        if llama_manager is not None and hasattr(llama_manager, "llm_call"):
            loop = asyncio.get_running_loop()
            smoke_prompt = (
                "Repeat exactly this text and nothing else:\n"
                "Testing, testing, 1, 2, 3."
            )
            response = await asyncio.wait_for(
                loop.run_in_executor(None, llama_manager.llm_call, smoke_prompt, 48),
                timeout=SMOKE_MODEL_TIMEOUT_SECONDS,
            )
        else:
            response = await asyncio.wait_for(
                functional_agent.handle_request(SMOKE_REQUEST),
                timeout=SMOKE_MODEL_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        error = f"timeout after {SMOKE_MODEL_TIMEOUT_SECONDS}s"
    except Exception as exc:  # pragma: no cover - depends on local model runtime
        error = str(exc)
    duration_ms = (time.perf_counter() - start) * 1000
    result = {
        "timestamp": _now_iso(),
        "mode": "model",
        "request": SMOKE_REQUEST,
        "expected": "Response contains 'testing' and '1, 2, 3'.",
        "response": response,
        "passed": error is None and _model_response_passes(response),
        "duration_ms": duration_ms,
        "error": error,
    }
    await _append_smoke_record(result)
    return result
