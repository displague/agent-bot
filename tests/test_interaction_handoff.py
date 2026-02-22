import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from interaction_processor import InteractionProcessor

@pytest.fixture(autouse=True)
def mock_model_load():
    with patch('llama_model_manager.LlamaModelManager.load_model'):
        yield

@pytest.mark.asyncio
async def test_interaction_processor_handoff():
    interaction_queue = asyncio.Queue()
    state = {"unprocessed_interactions": 1, "is_processing": False}
    llama_manager = MagicMock()
    log_manager = AsyncMock()
    index_manager = MagicMock()
    voice_loop = MagicMock()
    personaplex_manager = MagicMock()
    personaplex_manager.infer_async = AsyncMock()
    
    with patch('interaction_processor.FunctionalAgent', return_value=AsyncMock()) as mock_agent_class:
        processor = InteractionProcessor(
            interaction_queue, state, llama_manager, log_manager, index_manager,
            voice_loop=voice_loop, personaplex_manager=personaplex_manager
        )
        
        # Mock functional agent to return a response
        processor.functional_agent.handle_request.return_value = "Deep reasoning result"
        
        # Mock the internal filler trigger to avoid actual model load
        processor._trigger_verbal_filler = AsyncMock()
        
        # Queue an interaction
        interaction_queue.put_nowait({"input": "Hello", "audio_waveform": None})
        
        # Run one loop iteration by requesting stop after a short wait
        async def run_processor():
            # Process one item then stop
            task = asyncio.create_task(processor.start())
            while state["unprocessed_interactions"] > 0:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.2) # Allow time for final response speak
            processor.request_stop()
            await task
            
        await run_processor()
        
        # Verify filler was triggered
        processor._trigger_verbal_filler.assert_called_once()
        # Verify deep reasoning was triggered
        processor.functional_agent.handle_request.assert_called_with("Hello")
        # Verify final spoken response was triggered (via personaplex_manager.infer_async)
        personaplex_manager.infer_async.assert_called()
        assert personaplex_manager.infer_async.call_args.kwargs['text_prompt'] == "Deep reasoning result"

@pytest.mark.asyncio
async def test_pure_voice_transcription_handoff():
    interaction_queue = asyncio.Queue()
    state = {"unprocessed_interactions": 1, "is_processing": False}
    llama_manager = MagicMock()
    log_manager = AsyncMock()
    index_manager = MagicMock()
    
    with patch('interaction_processor.FunctionalAgent', return_value=AsyncMock()):
        processor = InteractionProcessor(
            interaction_queue, state, llama_manager, log_manager, index_manager
        )
        processor.functional_agent.handle_request.return_value = "I heard you"
        
        # Queue voice-only interaction
        mock_audio = np.zeros(16000, dtype=np.float32)
        interaction_queue.put_nowait({"input": "", "audio_waveform": mock_audio})
        
        # Mock transcribe_audio
        with patch('interaction_processor.transcribe_audio', new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "transcribed voice"
            
            async def run_processor():
                task = asyncio.create_task(processor.start())
                while state["unprocessed_interactions"] > 0:
                    await asyncio.sleep(0.1)
                processor.request_stop()
                await task
                
            await run_processor()
            
            # Verify transcription was called
            mock_transcribe.assert_called_once()
            # Verify it was passed to deep reasoning
            processor.functional_agent.handle_request.assert_called_with("transcribed voice")
