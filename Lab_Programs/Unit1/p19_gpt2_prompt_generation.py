"""Unit 1 - P19: Generate text from a prompt using the GPT-2 language model (user prompt)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import pipeline, set_seed

banner("P19 - GPT-2 text from a user prompt")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time, a robot in Chennai"
set_seed(1)
gen = pipeline("text-generation", model="gpt2")
for i, o in enumerate(gen(prompt, max_new_tokens=35, num_return_sequences=2, do_sample=True, pad_token_id=50256), 1):
    print(f"[{i}] {o['generated_text']}\n")
