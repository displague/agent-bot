import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import numpy as np
import json
from pathlib import Path

@pytest.mark.asyncio
async def test_interaction_processor_handoff():
    from interaction_processor import InteractionProcessor
    
    with patch('interaction_processor.SKIP_DEEP_REASONING', False):
        interaction_queue = asyncio.Queue()
        state = {"unprocessed_interactions": 1, "is_processing": False} 
        llama_manager = MagicMock()
        log_manager = AsyncMock()
        index_manager = MagicMock()
        voice_loop = MagicMock()
        personaplex_manager = MagicMock()
        personaplex_manager.infer = MagicMock()
        personaplex_manager.step_executor = MagicMock()

        with patch('interaction_processor.FunctionalAgent') as mock_agent_class:
            mock_agent_instance = mock_agent_class.return_value
            mock_agent_instance.handle_request = AsyncMock(return_value="Deep reasoning result")

            processor = InteractionProcessor(
                interaction_queue, state, llama_manager, log_manager, index_manager,
                voice_loop=voice_loop, personaplex_manager=personaplex_manager
            )

            # Mock the internal filler trigger to avoid actual model load
            processor._trigger_verbal_filler = AsyncMock()

            # Queue an interaction
            interaction_queue.put_nowait({"input": "Hello", "audio_waveform": None})

            # Run processor in background
            task = asyncio.create_task(processor.start())

            # Wait for item to be processed (unprocessed interaction count decremented)
            for _ in range(50):
                if state["unprocessed_interactions"] == 0:
                    break
                await asyncio.sleep(0.05)

            processor.request_stop()
            await task

            # Verify filler was triggered
            processor._trigger_verbal_filler.assert_called_once()       
            # Verify deep reasoning was triggered
            mock_agent_instance.handle_request.assert_called_with("Hello")


@pytest.mark.asyncio
async def test_pure_voice_transcription_handoff():
    from interaction_processor import InteractionProcessor
    import utils
    
    with patch('interaction_processor.SKIP_DEEP_REASONING', False):
        interaction_queue = asyncio.Queue()
        state = {"unprocessed_interactions": 1, "is_processing": False} 
        llama_manager = MagicMock()
        log_manager = AsyncMock()
        index_manager = MagicMock()

        with patch('interaction_processor.FunctionalAgent') as mock_agent_class:
            mock_agent_instance = mock_agent_class.return_value
            mock_agent_instance.handle_request = AsyncMock(return_value="I heard you")

            processor = InteractionProcessor(
                interaction_queue, state, llama_manager, log_manager, index_manager
            )

            # Queue voice-only interaction
            mock_audio = np.zeros(16000, dtype=np.float32)
            interaction_queue.put_nowait({"input": "", "audio_waveform": mock_audio})

            # Patch transcribe_audio specifically for this test
            with patch('interaction_processor.transcribe_audio', new_callable=AsyncMock) as mock_trans:
                mock_trans.return_value = "transcribed voice"

                task = asyncio.create_task(processor.start())
                for _ in range(50):
                    if state["unprocessed_interactions"] == 0:
                        break
                    await asyncio.sleep(0.05)
                processor.request_stop()
                await task

                # Verify transcription was called
                mock_trans.assert_called_once()
                # Verify it was passed to deep reasoning
                mock_agent_instance.handle_request.assert_called_with("transcribed voice")
