import whisper

print("=" * 40)
print("SPEECH TO TEXT")
print("=" * 40)

# Load Whisper model
model = whisper.load_model("base")

# Audio file
audio_file = "sample.wav"

# Convert speech to text
result = model.transcribe(audio_file)

text = result["text"].strip()

print("\nTranscribed Text:")
print(text)

# Save for text emotion analysis
with open("transcribed_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("\nTranscribed text saved to transcribed_text.txt")