# ============================================================
# GEMINI API KEY
# ============================================================

API_KEY = "AQ.Ab8RN6JM8LQl_JNE33O_GoQNBC5dHHXpRzg5Z26IFQlUVNwB-w"


# ============================================================
# IMPORTS
# ============================================================

import os
import json
from datetime import datetime

import whisper
import torchaudio as ta
from transformers import pipeline
from google import genai
from chatterbox.tts import ChatterboxTTS


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_FILE = "sad.wav"              # Child input
PARENT_VOICE = "happy.wav"          # Parent reference voice
HISTORY_FILE = "conversation_history.json"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# SAVE CONVERSATION
# ============================================================

def save_conversation(
    user_text,
    emotion,
    confidence,
    ai_reply
):

    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recognized_text": user_text,
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "ai_reply": ai_reply
    }

    if os.path.exists(HISTORY_FILE):

        try:
            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if not isinstance(data, list):
                data = []

        except Exception:

            data = []

    else:

        data = []

    data.append(item)

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper...")

whisper_model = whisper.load_model("base")


# ============================================================
# LOAD TEXT EMOTION MODEL
# ============================================================

print("Loading Emotion Model...")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


# ============================================================
# LOAD CHATTERBOX
# ============================================================

print("Loading Chatterbox...")

tts = ChatterboxTTS.from_pretrained(
    device="cpu"
)


# ============================================================
# SPEECH TO TEXT
# ============================================================

print("\nTranscribing...")

result = whisper_model.transcribe(
    AUDIO_FILE
)

transcript = result["text"].strip()

print("\n==================== TRANSCRIPTION ====================")

print(transcript)


# ============================================================
# TEXT EMOTION DETECTION
# ============================================================

print("\n==================== EMOTION ANALYSIS ====================")

emotion_results = emotion_classifier(
    transcript
)


# ============================================================
# SORT EMOTIONS
# ============================================================

emotion_results = sorted(
    emotion_results[0],
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# GET BEST EMOTION
# ============================================================

emotion = emotion_results[0]["label"]

confidence = emotion_results[0]["score"]


# ============================================================
# DISPLAY ALL EMOTION SCORES
# ============================================================

print("\nEmotion probabilities:")

for result in emotion_results:

    label = result["label"]

    score = result["score"]

    print(
        f"{label:10s} : {score * 100:.2f}%"
    )


# ============================================================
# FINAL EMOTION
# ============================================================

print(
    "\n----------------------------------------------------------"
)

print(
    f"FINAL DETECTED EMOTION : {emotion}"
)

print(
    f"CONFIDENCE             : {confidence * 100:.2f}%"
)

print(
    "----------------------------------------------------------"
)


# ============================================================
# GEMINI PARENT RESPONSE
# ============================================================

print("\nGenerating AI parent response...")


prompt = f"""
You are a caring and supportive parent.

Child's message:
{transcript}

The system detected the child's emotional state as:
{emotion}

Respond naturally like a caring parent.

Rules:
- Reply in 2-4 sentences.
- Be warm and supportive.
- Respond appropriately to the child's situation.
- Do not mention the emotion detection system.
- Do not say things like "I detected that you are sad".
- Do not mention AI or machine learning.
"""


# ============================================================
# CALL GEMINI
# ============================================================

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)


# ============================================================
# GET GEMINI RESPONSE
# ============================================================

ai_reply = response.text.strip()


print(
    "\n==================== AI REPLY ===================="
)

print(ai_reply)


# ============================================================
# SAVE CONVERSATION
# ============================================================

save_conversation(
    transcript,
    emotion,
    confidence,
    ai_reply
)

print(
    "\nConversation saved to:"
)

print(HISTORY_FILE)


# ============================================================
# GENERATE PARENT VOICE
# ============================================================

print(
    "\nGenerating parent voice..."
)


wav = tts.generate(
    ai_reply,
    audio_prompt_path=PARENT_VOICE
)


# ============================================================
# SAVE AUDIO
# ============================================================

ta.save(
    "reply.wav",
    wav,
    tts.sr
)


print(
    "\nParent voice saved as: reply.wav"
)


# ============================================================
# PLAY AUDIO
# ============================================================

try:

    os.startfile(
        "reply.wav"
    )

except Exception:

    pass


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n=========================================================="
)

print(
    "                     FINAL RESULT"
)

print(
    "=========================================================="
)


print(
    "\nRecognized Text:"
)

print(
    transcript
)


print(
    "\nDetected Emotion:"
)

print(
    f"{emotion} ({confidence * 100:.2f}%)"
)


print(
    "\nAI Parent Reply:"
)

print(
    ai_reply
)


print(
    "\nOutput Audio:"
)

print(
    "reply.wav"
)


print(
    "\n=========================================================="
)

print(
    "Finished."
)