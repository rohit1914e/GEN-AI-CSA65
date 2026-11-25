"""Unit 1 - P20: Generate and compare responses for multiple prompts using a pre-trained LLM."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import pipeline, set_seed

banner("P20 - Responses of a pre-trained LLM (GPT-2) for multiple prompts")
set_seed(3)
gen = pipeline("text-generation", model="gpt2")
prompts = ["Machine learning is", "The best way to prepare for exams is", "In the year 2050, cars will"]
for p in prompts:
    t = gen(p, max_new_tokens=30, do_sample=True, top_p=0.9, pad_token_id=50256)[0]["generated_text"]
    print(f"Prompt: {p}\n  -> {t}\n  words={len(t.split())}\n")
