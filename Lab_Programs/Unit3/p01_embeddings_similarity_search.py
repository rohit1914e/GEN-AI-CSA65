"""Unit 3 - P1: Generate text embeddings and perform semantic similarity search."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, EMBED_MODEL  # noqa: E402

banner("U3 P1 - Text embeddings + semantic similarity search (" + EMBED_MODEL + ")")
sentences = ["The cat sat on the mat.", "A kitten is resting on a rug.", "Stock prices fell sharply today.",
             "The football match ended in a draw.", "Investors lost money in the market crash.", "Dogs love playing fetch."]
vecs = embed(sentences)
print("Embedding matrix shape:", vecs.shape, "| first 5 dims of S1:", vecs[0][:5].round(3).tolist())
for q in ["a small cat lying on carpet", "financial losses in shares"]:
    sims = vecs @ embed([q], query=True)[0]
    print(f"\nQuery: {q}")
    for i in sims.argsort()[::-1][:3]:
        print(f"  {sims[i]:.3f}  {sentences[i]}")
