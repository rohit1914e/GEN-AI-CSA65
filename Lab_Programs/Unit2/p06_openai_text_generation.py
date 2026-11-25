"""Unit 2 - P6: Text Generation using OpenAI LLM API - connect to an LLM using the OpenAI API and generate text from a user prompt.

Needs OPENAI_API_KEY in .env (pip install openai). Missing key / package is handled gracefully.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402

banner("U2 P6 - Text generation with the OpenAI API")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain what a large language model is in 3 sentences."
print("Prompt:", prompt)
key = os.getenv("OPENAI_API_KEY")
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
if not key or OpenAI is None:
    print("OpenAI API not run: " + ("'openai' package not installed (pip install openai). " if OpenAI is None else "") +
          ("OPENAI_API_KEY is not set in .env." if not key else ""))
    print("Request that would be sent: client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content': prompt}])")
else:
    client = OpenAI(api_key=key)
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=200)
        print("Response:", r.choices[0].message.content)
    except Exception as e:
        print("OpenAI API error:", type(e).__name__, e)
