import subprocess

print("="*60)
print("MULTIMODAL EMOTION RECOGNITION")
print("="*60)

print("\n1. Speech To Text")
subprocess.run(["python", "speech_test.py"])

print("\n2. Text Emotion")
subprocess.run(["python", "text_emotion.py"])

print("\n3. Speech Emotion")
subprocess.run(["python", "emotion.py"])

# Read Results
with open("text_result.txt") as f:
    text_emotion, text_conf = f.read().split(",")

with open("speech_result.txt") as f:
    speech_emotion, speech_conf = f.read().split(",")

text_conf = float(text_conf)
speech_conf = float(speech_conf)

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)

print(f"Text Emotion   : {text_emotion} ({text_conf:.2f}%)")
print(f"Speech Emotion : {speech_emotion} ({speech_conf:.2f}%)")

if text_emotion.lower() == speech_emotion.lower():
    final = text_emotion
else:
    if speech_conf > text_conf:
        final = speech_emotion
    else:
        final = text_emotion

print(f"\nOverall Emotion : {final}")