import whisper
from transformers import pipeline
from google import genai
import asyncio
import edge_tts
import json
import os
from datetime import datetime

# ==========================================================
# Save Conversation Function
# ==========================================================
def save_conversation(user_text, emotion, confidence, ai_reply):
    file_name = "conversation_history.json"

    conversation = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recognized_text": user_text,
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "ai_reply": ai_reply
    }

    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(conversation)

    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("\nConversation saved to conversation_history.json")


# ==========================================================
# Edge-TTS Function
# ==========================================================
async def speak(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )

    await communicate.save("reply.mp3")


# ==========================================================
# Step 1: Configure Gemini Client
# ==========================================================
API_KEY = "API"

client = genai.Client(api_key=API_KEY)


# ==========================================================
# Step 2: Load Whisper Model
# ==========================================================
print("\nLoading Whisper model...")
whisper_model = whisper.load_model("base")


# ==========================================================
# Step 3: Load Emotion Detection Model
# ==========================================================
print("Loading Emotion Detection model...")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)


# ==========================================================
# Step 4: Input Audio File
# ==========================================================
audio_file = "happy.wav"


# ==========================================================
# Step 5: Speech to Text
# ==========================================================
print("\nConverting Speech to Text...")

result = whisper_model.transcribe(audio_file)

transcript = result["text"].strip()

print("\nRecognized Text:")
print(transcript)


# ==========================================================
# Step 6: Emotion Detection
# ==========================================================
print("\nDetecting Emotion...")

emotion_result = emotion_classifier(transcript)

emotion = emotion_result[0][0]["label"]
confidence = emotion_result[0][0]["score"]


# ==========================================================
# Step 7: Generate AI Reply
# ==========================================================
prompt = f"""
You are an intelligent AI assistant that replies like a caring parent.

User Message:
{transcript}

Detected Emotion:
{emotion}

Instructions:

1. First understand the user's message and the situation.
2. Reply according to the situation, not only the detected emotion.
3. Use the detected emotion only to adjust the tone of your reply.
4. Speak naturally like a caring parent.
5. Be warm, supportive, encouraging, and practical.
6. If the user achieves something, congratulate and motivate them.
7. If the user is sad or fails, comfort them and encourage them.
8. If the user asks a question, answer it clearly.
9. Keep the reply between 2 and 4 sentences.
10. Do NOT mention the detected emotion.
11. Do NOT say "As an AI..." or "I detected your emotion."
12. Do NOT invent or add local words such as "da", "dei", "pa", "ma", "kanna", etc.
13. Only use local or repeated words if they are explicitly provided as part of the parent's learned speaking style.
14. If no parent speaking style is provided, use standard, natural English.

Generate only the reply.
"""
try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    ai_reply = response.text

except Exception as e:
    ai_reply = f"Error generating AI reply: {e}"


# ==========================================================
# Step 8: Display Result
# ==========================================================
print("\n==================== RESULT ====================")

print("\nRecognized Text:")
print(transcript)

print("\nDetected Emotion:")
print(emotion)

print(f"\nConfidence: {confidence:.2%}")

print("\nAI Reply:")
print(ai_reply)


# ==========================================================
# Step 9: Save Conversation
# ==========================================================
save_conversation(
    transcript,
    emotion,
    confidence,
    ai_reply
)


# ==========================================================
# Step 10: Convert AI Reply to Speech
# ==========================================================
print("\nConverting AI Reply to Speech...")

try:
    asyncio.run(speak(ai_reply))

    print("Audio saved as reply.mp3")

    # Automatically play audio (Windows)
    os.startfile("reply.mp3")

except Exception as e:
    print("Speech Generation Error:", e)

print("\n==================== FINISHED ====================")
