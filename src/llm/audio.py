import torch
from transformers import pipeline


class AudioTranscriber:
    def __init__(self):
        # Usamos "openai/whisper-tiny" o "whisper-base" para que sea rápido en CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=self.device,
        )

    def transcribe(self, audio_path):
        return self.transcriber(audio_path)
