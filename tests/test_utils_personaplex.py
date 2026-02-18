import json
from pathlib import Path
from types import SimpleNamespace

import utils


def test_decode_output_tokens_filters_special_tokens(tmp_path):
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(["BOS", " hello", " world", "PAD", "EOS"]),
        encoding="utf-8",
    )

    text = utils._decode_output_tokens(str(output_path))

    assert text == "hello world"


def test_run_personaplex_offline_returns_generated_text(monkeypatch, tmp_path):
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    output_text = tmp_path / "text.json"
    input_wav.write_bytes(b"fake")

    captured = {}

    def fake_run(command, check, text, capture_output, timeout, env):
        captured["command"] = command
        Path(output_text).write_text(json.dumps([" hi", " there"]), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    result = utils.run_personaplex_offline(
        str(input_wav),
        str(output_wav),
        output_text=str(output_text),
        voice_prompt="C:/voices/NATF2.pt",
        voice_prompt_dir="C:/voices",
        text_prompt="You enjoy having a good conversation.",
    )

    assert "--voice-prompt" in captured["command"]
    assert "NATF2.pt" in captured["command"]
    assert "--voice-prompt-dir" in captured["command"]
    assert result["generated_text"] == "hi there"
