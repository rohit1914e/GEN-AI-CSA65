"""Unit 1 - P13: Generate text using a pre-trained GPT-2 model (model + tokenizer, greedy vs sampling)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

banner("P13 - GPT-2 generation: greedy vs sampling")
tok = GPT2Tokenizer.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2")
ids = tok("The future of renewable energy in India", return_tensors="pt").input_ids
torch.manual_seed(7)
greedy = model.generate(ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
sample = model.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
print("Greedy  :", tok.decode(greedy[0], skip_special_tokens=True))
print("Sampling:", tok.decode(sample[0], skip_special_tokens=True))
