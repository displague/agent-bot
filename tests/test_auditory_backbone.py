import asyncio
import numpy as np
import pytest
from utils import AudioMultiplexer, RollingAudioBuffer

@pytest.mark.asyncio
async def test_audio_multiplexer_broadcast():
    # Mock chunk
    sample_rate = 16000
    chunk_seconds = 0.1
    chunk_size = int(sample_rate * chunk_seconds)
    mock_chunk = np.random.rand(chunk_size).astype(np.float32)
    
    mux = AudioMultiplexer(sample_rate=sample_rate, chunk_seconds=chunk_seconds)
    q1 = mux.subscribe()
    q2 = mux.subscribe()
    
    # Manually push a chunk into subscribers (bypassing the capture thread for unit test)
    with mux._lock:
        for q in mux.subscribers:
            q.put_nowait(mock_chunk)
            
    # Verify both queues received the chunk
    received1 = await q1.get()
    received2 = await q2.get()
    
    assert np.array_equal(received1, mock_chunk)
    assert np.array_equal(received2, mock_chunk)
    
    mux.unsubscribe(q1)
    assert len(mux.subscribers) == 1
    assert q2 in mux.subscribers

@pytest.mark.asyncio
async def test_rolling_audio_buffer():
    sample_rate = 16000
    chunk_seconds = 0.4
    mux = AudioMultiplexer(sample_rate=sample_rate, chunk_seconds=chunk_seconds)
    
    # 2 second buffer
    buffer = RollingAudioBuffer(mux, seconds=2.0)
    await buffer.start()
    
    # Push 3 chunks (1.2s total)
    chunks = [np.full(int(sample_rate * chunk_seconds), i, dtype=np.float32) for i in range(3)]
    for c in chunks:
        await buffer.queue.put(c)
        
    # Give it a tiny bit of time to process the queue
    await asyncio.sleep(0.5)
    
    # Retrieve last 0.8s (exactly 2 chunks)
    recent = buffer.get_recent(seconds=0.8)
    # Expected size is 0.8 * sample_rate
    assert len(recent) == int(sample_rate * 0.8)
    
    # The last chunk should be all 2s
    assert np.all(recent[-10:] == 2.0)
    
    await buffer.stop()
