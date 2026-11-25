"""Unit 1 - P8: Tokenization using BERT Tokenizer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import BertTokenizer

banner("P8 - Tokenization using the BERT tokenizer")
tok = BertTokenizer.from_pretrained("bert-base-uncased")
s = "Tokenization splits unbelievably long words into sub-words."
print("Sentence :", s)
print("Tokens   :", tok.tokenize(s))
enc = tok(s)
print("Input IDs:", enc["input_ids"])
print("Decoded  :", tok.decode(enc["input_ids"]))
print("Vocabulary size:", tok.vocab_size)
