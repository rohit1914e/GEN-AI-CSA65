"""Unit 1 - P18: Compare contextual embeddings generated for two semantically similar sentences."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import torch
from transformers import AutoTokenizer, AutoModel

banner("P18 - Similar vs different sentences (BERT mean-pooled embeddings)")
tok = AutoTokenizer.from_pretrained("bert-base-uncased"); model = AutoModel.from_pretrained("bert-base-uncased")


def embed(s):
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    return h.mean(0)


s1, s2, s3 = "The weather is lovely today.", "It is a beautiful sunny day.", "The stock market crashed yesterday."
cos = lambda x, y: torch.nn.functional.cosine_similarity(embed(x), embed(y), dim=0).item()
print(f"S1: {s1}\nS2: {s2}\nS3: {s3}")
print(f"cos(S1, S2) similar   = {cos(s1, s2):.3f}")
print(f"cos(S1, S3) different = {cos(s1, s3):.3f}")
