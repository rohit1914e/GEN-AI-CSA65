"""Unit 1 - P11: Load the BERT tokenizer and tokenize a sentence."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import AutoTokenizer

banner("P11 - Load the BERT tokenizer and tokenize a sentence")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
s = "Saveetha students are learning large language models."
print("Sentence:", s)
print("Tokens  :", tok.tokenize(s))
print("With special tokens:", tok.convert_ids_to_tokens(tok(s)["input_ids"]))
