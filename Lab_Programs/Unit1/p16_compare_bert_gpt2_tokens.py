"""Unit 1 - P16: Compare the tokenization outputs of BERT and GPT-2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import AutoTokenizer

banner("P16 - BERT (WordPiece) vs GPT-2 (byte-level BPE) tokenization")
bert = AutoTokenizer.from_pretrained("bert-base-uncased"); gpt2 = AutoTokenizer.from_pretrained("gpt2")
for s in ["Tokenization is fundamental for NLP.", "I'm learning Transformers in 2026!"]:
    b, g = bert.tokenize(s), gpt2.tokenize(s)
    print("Sentence:", s)
    print(f"  BERT  ({len(b)} tokens): {b}")
    print(f"  GPT-2 ({len(g)} tokens): {g}")
print("Notes: BERT lower-cases and marks sub-words with ##; GPT-2 keeps case and marks a leading space with G-dot.")
