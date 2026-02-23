import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.fixture(autouse=True)
def mock_all_heavy_components(request):
    """Globally mock all components that trigger heavy loading or subprocesses, 
    unless the test is marked with @pytest.mark.skip_heavy_mock."""
    if "skip_heavy_mock" in request.keywords:
        yield
        return

    with (
        patch('llama_model_manager.LlamaModelManager.load_model'),
        patch('utils.PersonaPlexManager.load'),
        patch('utils.run_personaplex_offline', new_callable=AsyncMock) as mock_offline,
        patch('utils.capture_microphone_chunk', return_value=None),
        patch('utils.capture_screen', return_value=MagicMock())
    ):
        mock_offline.return_value = {"generated_text": "mock response", "output_wav": "mock.wav", "stdout": "mock"}
        yield
