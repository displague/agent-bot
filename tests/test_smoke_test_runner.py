import asyncio
import json

import smoke_test_runner as smoke


def test_run_deterministic_smoke_writes_expected_record(tmp_path, monkeypatch):
    log_path = tmp_path / "smoke.jsonl"
    monkeypatch.setattr(smoke, "SMOKE_LOG_PATH", str(log_path))

    result = asyncio.run(smoke.run_deterministic_smoke())

    assert result["mode"] == "deterministic"
    assert result["passed"] is True
    assert result["request"] == smoke.SMOKE_REQUEST
    assert result["response"] == smoke.SMOKE_EXPECTED_RESPONSE
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["passed"] is True


class _FakeAgent:
    async def handle_request(self, prompt):
        return "Testing, testing, 1, 2, 3."


def test_run_model_smoke_passes_with_matching_phrase(tmp_path, monkeypatch):
    log_path = tmp_path / "smoke.jsonl"
    monkeypatch.setattr(smoke, "SMOKE_LOG_PATH", str(log_path))

    result = asyncio.run(smoke.run_model_smoke(_FakeAgent()))

    assert result["mode"] == "model"
    assert result["passed"] is True
    summary = smoke.summarize_smoke_result(result)
    assert summary.startswith("SMOKE model: PASS")
