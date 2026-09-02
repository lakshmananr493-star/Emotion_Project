import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# ===============================
# Load Audio
# ===============================

audio_file = "sample.wav"

audio, sr = librosa.load(audio_file, sr=16000)

print("="*60)
print("AUDIO INFORMATION")
print("="*60)

print("Sample Rate :", sr)
print("Duration    :", round(librosa.get_duration(y=audio, sr=sr),2), "seconds")
print("Samples     :", len(audio))

# ===============================
# RMS Energy
# ===============================

rms = np.sqrt(np.mean(audio**2))
print("\nAverage RMS Energy :", rms)

# ===============================
# Pitch Estimation
# ===============================

print("\nEstimating Pitch...")

f0, voiced_flag, voiced_prob = librosa.pyin(
    audio,
    fmin=75,
    fmax=400
)

avg_pitch = np.nanmean(f0)

print("Average Pitch :", round(avg_pitch,2), "Hz")

# ===============================
# Load WavLM
# ===============================

print("\nLoading WavLM Model...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-large"
)

model = WavLMModel.from_pretrained(
    "microsoft/wavlm-large"
)

print("WavLM Loaded Successfully!")

# ===============================
# Feature Extraction
# ===============================

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

features = outputs.last_hidden_state

print("\nWavLM Feature Shape :", features.shape)

print("\nFirst 20 Feature Values")

print(features[0,0,:20])

print("\nMean Feature :", features.mean().item())
print("Std Feature  :", features.std().item())

# ===============================
# Save Features
# ===============================

np.save("features.npy", features.numpy())

print("\nFeatures saved as features.npy")

# ===============================
# Waveform
# ===============================

plt.figure(figsize=(12,4))

plt.plot(audio)

plt.title("Audio Waveform")

plt.xlabel("Samples")

plt.ylabel("Amplitude")

plt.grid(True)

plt.show()

# ===============================
# Spectrogram
# ===============================

D = librosa.amplitude_to_db(
    np.abs(librosa.stft(audio)),
    ref=np.max
)

plt.figure(figsize=(12,4))

librosa.display.specshow(
    D,
    sr=sr,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar()

plt.title("Spectrogram")

plt.show()

# ===============================
# WavLM Heatmap
# ===============================

feature_matrix = features[0].numpy()

plt.figure(figsize=(14,6))

plt.imshow(
    feature_matrix.T,
    aspect='auto',
    origin='lower'
)

plt.title("WavLM Feature Heatmap")

plt.xlabel("Time Frames")

plt.ylabel("Feature Dimension")

plt.colorbar()

plt.show()

print("\nAnalysis Completed Successfully!")