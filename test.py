import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")

wav = model.generate(
    "Hello! This is my first cloned voice."
)

ta.save("output.wav", wav, model.sr)

print("Done!")