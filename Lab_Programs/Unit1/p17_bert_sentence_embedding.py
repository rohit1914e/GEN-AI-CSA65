"""Unit 1 - P17: Generate contextual embeddings for a sentence using BERT (same word, different contexts)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import torch
from transformers import AutoTokenizer, AutoModel

banner("P17 - Contextual embeddings: 'bank' in two contexts")
tok = AutoTokenizer.from_pretrained("bert-base-uncased"); model = AutoModel.from_pretrained("bert-base-uncased")


def word_vec(sentence, word):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    return h[tok.convert_ids_to_tokens(enc["input_ids"][0]).index(word)]


a = word_vec("I deposited money in the bank.", "bank")
b = word_vec("We sat on the river bank.", "bank")
c = word_vec("The bank approved my loan.", "bank")
cos = torch.nn.functional.cosine_similarity
print("Embedding size:", a.shape[0])
print(f"money-bank vs river-bank : {cos(a, b, dim=0):.3f}")
print(f"money-bank vs loan-bank  : {cos(a, c, dim=0):.3f}  (higher -> same meaning)")
