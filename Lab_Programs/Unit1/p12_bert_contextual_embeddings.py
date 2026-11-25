"""Unit 1 - P12: Load a pre-trained BERT model and generate contextual embeddings."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import torch
from transformers import AutoTokenizer, AutoModel

banner("P12 - Contextual embeddings from BERT")
tok = AutoTokenizer.from_pretrained("bert-base-uncased"); model = AutoModel.from_pretrained("bert-base-uncased")
s = "Large language models understand context."
with torch.no_grad():
    out = model(**tok(s, return_tensors="pt"))
h = out.last_hidden_state[0]
print("Sentence:", s)
print("last_hidden_state shape:", tuple(out.last_hidden_state.shape), "(batch, tokens, hidden size)")
for t, v in zip(tok.convert_ids_to_tokens(tok(s)["input_ids"]), h):
    print(f"  {t:12s} first 5 dims: {[round(x, 3) for x in v[:5].tolist()]}")
print("Sentence embedding (mean pooling) first 5 dims:", [round(x, 3) for x in h.mean(0)[:5].tolist()])
