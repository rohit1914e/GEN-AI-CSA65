"""Unit 1 - P15: Tokenize a sentence using the BERT tokenizer and display the generated tokens and token IDs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import AutoTokenizer

banner("P15 - BERT tokens and token IDs")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
s = "ChatGPT generates human-like responses."
enc = tok(s)
print(f"{'Token':15s} Token ID")
for t, i in zip(tok.convert_ids_to_tokens(enc["input_ids"]), enc["input_ids"]):
    print(f"{t:15s} {i}")
print("Attention mask:", enc["attention_mask"])
