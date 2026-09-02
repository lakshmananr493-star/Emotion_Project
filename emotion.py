from transformers import pipeline

print("="*40)
print("SPEECH EMOTION")
print("="*40)

classifier = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

result = classifier("sample.wav")

emotion = result[0]["label"]
confidence = result[0]["score"] * 100

print("Detected Emotion :", emotion)
print("Confidence       : {:.2f}%".format(confidence))

with open("speech_result.txt", "w") as f:
    f.write(f"{emotion},{confidence}")