import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

print("Loading WavLM...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-large"
)
model = WavLMModel.from_pretrained(
    "microsoft/wavlm-large"
)

# Load audio directly (16 kHz mono)
waveform, sample_rate = librosa.load("sample.wav", sr=16000)

inputs = feature_extractor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

features = outputs.last_hidden_state

print("Feature shape:", features.shape)