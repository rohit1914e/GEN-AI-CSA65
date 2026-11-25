"""Unit 2 - P8: Text Generation using the Hugging Face Inference API (needs HF_TOKEN in .env)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import requests

banner("U2 P8 - Text generation with the Hugging Face Inference API")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Write one sentence about the benefits of open-source AI models."
token = os.getenv("HF_TOKEN")
url = "https://router.huggingface.co/v1/chat/completions"
payload = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": prompt}], "max_tokens": 120}
print("Prompt:", prompt)
if not token:
    print("HF_TOKEN not set in .env -> request not sent. Endpoint:", url)
else:
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=60)
        if r.status_code == 200:
            print("Response:", r.json()["choices"][0]["message"]["content"])
        else:
            print(f"Hugging Face API returned HTTP {r.status_code}: {r.text[:200]}")
    except requests.RequestException as e:
        print("Network error:", e)
