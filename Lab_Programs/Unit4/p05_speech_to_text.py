"""Unit 4 - P5: Speech-to-Text application that converts a spoken engineering-related query into text using a
pre-trained AI model (OpenAI Whisper 'base' via faster-whisper). Uses outputs/U4_P06_speech.wav or any file given."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from faster_whisper import WhisperModel

banner("U4 P5 - Speech-to-Text with Whisper")
audio = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "U4_P06_speech.wav")
if not os.path.exists(audio):
    from p06_text_to_speech import tts
    tts("What is the maximum load capacity of the hydraulic press?", audio)
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, language="en")
print(f"Audio: {os.path.basename(audio)} | duration {info.duration:.1f}s | detected language {info.language}")
for s in segments:
    print(f"[{s.start:5.1f}s - {s.end:5.1f}s] {s.text.strip()}")
