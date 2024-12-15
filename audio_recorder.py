# audio_recorder.py

import threading
import sounddevice as sd
import torch
import logging

logger = logging.getLogger("autonomous_system.audio_recorder")


class AudioRecorder(threading.Thread):
    def __init__(self, duration=5, sample_rate=16000):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_waveform = None
        self.logger = logging.getLogger("autonomous_system.audio_recorder")

    def run(self):
        """Records audio for the specified duration."""
        self.logger.debug("Recording audio...")
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
        )
        sd.wait()
        self.audio_waveform = torch.from_numpy(audio_data.flatten())
        self.logger.debug("Audio recording completed.")
