"""Unit 1 - P9: Generate text using a pre-trained GPT-2 model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import pipeline, set_seed

banner("P9 - Text generation with GPT-2")
set_seed(42)
gen = pipeline("text-generation", model="gpt2")
prompt = "Artificial intelligence will change engineering education by"
out = gen(prompt, max_new_tokens=40, do_sample=True, top_k=50, pad_token_id=50256)[0]["generated_text"]
print("Prompt   :", prompt); print("Generated:", out)
