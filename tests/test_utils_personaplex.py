import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import utils


def test_decode_output_tokens_filters_special_tokens(tmp_path):
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(["BOS", " hello", " world", "PAD", "EOS"]),
        encoding="utf-8",
    )

    text = utils._decode_output_tokens(str(output_path))

    assert text == "hello world"


@pytest.mark.skip_heavy_mock
@pytest.mark.asyncio
async def test_run_personaplex_offline_returns_generated_text(monkeypatch, tmp_path):
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    output_text = tmp_path / "text.json"
    input_wav.write_bytes(b"fake")

    # Mock the in-process inference function
    async def mock_moshi_run_inference(*args, **kwargs):
        # The caller expects output_text file to exist
        Path(output_text).write_text(json.dumps([" hi", " there"]), encoding="utf-8")
        return None

    # Patch the in-process path which is tried first
    with patch('utils.moshi_run_inference', new=mock_moshi_run_inference), \
         patch('utils._ensure_voice_prompt_exists', return_value="C:/voices/NATF2.pt"):
        
        async def mock_to_thread(func, *args, **kwargs):
            return await func(*args, **kwargs)
            
        with patch('asyncio.to_thread', side_effect=mock_to_thread):
            result = await utils.run_personaplex_offline(
                str(input_wav),
                str(output_wav),
                output_text=str(output_text),
                voice_prompt="C:/voices/NATF2.pt",
                voice_prompt_dir="C:/voices",
                text_prompt="You enjoy having a good conversation.",
            )

            assert result["generated_text"] == "hi there"
            assert result["stdout"] == "in-process execution"
