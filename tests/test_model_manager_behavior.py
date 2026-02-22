import asyncio
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from llama_model_manager import LlamaModelManager

@pytest.fixture(autouse=True)
def mock_model_load():
    with patch('llama_model_manager.LlamaModelManager.load_model'):
        yield

@pytest.mark.asyncio
async def test_model_manager_tool_feedback_loop():
    # Mock manager dependencies
    manager = LlamaModelManager(model_path=None)
    manager.backend = "llama_cpp"
    manager.llm_executor = "mock_executor"
    
    # Mock tool that returns raw audio
    manager.available_functions["inspect_audio_snippet"] = MagicMock(return_value=np.zeros(16000, dtype=np.float32))
    
    # First call to llm_call returns the tool call JSON
    # Second call returns the final text
    with patch.object(manager, 'llm_call') as mock_llm_call:
        mock_llm_call.side_effect = [
            '{"name": "inspect_audio_snippet", "arguments": {"seconds": 5, "return_raw": true}}',
            "I hear silence."
        ]
        
        # We need to mock run_in_executor to call the function directly
        async def mock_run_in_executor(executor, func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_get_loop.return_value = mock_loop
            
            result = await manager.run_phase("Execution", "Check audio", "Notes")
            
            # Verify tool was called
            manager.available_functions["inspect_audio_snippet"].assert_called_once()
            # Verify llm_call was re-invoked with audio
            assert mock_llm_call.call_count == 2
            # Check the second call specifically
            args, kwargs = mock_llm_call.call_args_list[1]
            # audio is the 3rd positional arg: llm_call(prompt, max_tokens, audio, image)
            assert args[2] is not None
            assert result == "I hear silence."

@pytest.mark.asyncio
async def test_conversation_alternation_logic():
    manager = LlamaModelManager(model_path=None)
    manager.llm_context = [
        "User: Hello",
        "User: How are you?", # Consecutive user
        "Assistant: I am fine",
        "Assistant: Thanks for asking" # Consecutive assistant
    ]
    
    # We test the logic inside llm_call by patching apply_chat_template
    manager.backend = "transformers"
    manager.hf_model = MagicMock()
    manager.hf_tokenizer = MagicMock()
    manager.hf_tokenizer.apply_chat_template = MagicMock(return_value="templated")
    manager.hf_model.generate = MagicMock(return_value=[[0]]) # Dummy token
    manager.hf_tokenizer.decode = MagicMock(return_value="Response")
    
    # Call llm_call
    manager.llm_call("New prompt")
    
    # Inspect the messages passed to apply_chat_template
    args, kwargs = manager.hf_tokenizer.apply_chat_template.call_args
    messages = args[0]
    
    # Should be: User (merged), Assistant (merged), User (current prompt)
    assert len(messages) == 3
    assert messages[0]['role'] == 'user'
    assert """Hello
How are you?""" in messages[0]['content']
    assert messages[1]['role'] == 'assistant'
    assert """I am fine
Thanks for asking""" in messages[1]['content']
    assert messages[2]['role'] == 'user'
    assert messages[2]['content'] == "New prompt"
