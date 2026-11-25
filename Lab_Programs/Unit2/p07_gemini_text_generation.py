"""Unit 2 - P7: Text Generation using Google Gemini LLM API - responses for different user prompts."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402

banner("U2 P7 - Text generation with the Google Gemini API")
prompts = sys.argv[1:] or ["Write a haiku about monsoon in Chennai.", "Give 3 tips to learn Python faster.",
                           "Explain recursion to a 10-year-old in 2 lines."]
for p in prompts:
    print(f"\nPrompt  : {p}\nResponse: {gemini(p)}")
