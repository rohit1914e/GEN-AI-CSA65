"""Unit 4 - P6: Text-to-Speech application that converts engineering-related text into natural-sounding speech
using a pre-trained AI model (Google Gemini TTS)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import wave
from google import genai
from google.genai import types


def tts(text, path, voice="Kore"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    r = client.models.generate_content(model="gemini-3.8-flash-tts", contents=text,
        config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
    data = r.candidates[0].content.parts[0].inline_data.data
    if data[:4] == b"RIFF":
        open(path, "wb").write(data)
    else:                                   # raw 16-bit PCM, 24 kHz mono -> wrap in a WAV header
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


if __name__ == "__main__":
    banner("U4 P6 - Text-to-Speech (Gemini TTS)")
    text = ("The load test on the steel bridge was successful. Maximum deflection was twelve millimetres, "
            "which is within the permitted limit.")
    path = os.path.join(OUT, "U4_P06_speech.wav")
    secs = tts(text, path)
    print(f"Text : {text}\nAudio: outputs/{os.path.basename(path)}  ({secs:.1f} seconds, 24 kHz mono WAV)")
