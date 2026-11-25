"""Unit 3 - P2: Semantic search system using cosine similarity between document and query embeddings (NumPy only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text  # noqa: E402
import numpy as np

banner("U3 P2 - Cosine-similarity semantic search over academic documents")
docs = [d for d in load_text("academic_documents.txt").split("\n\n") if d.strip()]
D = embed(docs)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


for q in ["How do computers decide which process runs next?", "secure communication with keys", "grouping unlabeled data"]:
    qv = embed([q], query=True)[0]
    scores = sorted(((cosine(qv, d), doc.splitlines()[0]) for d, doc in zip(D, docs)), reverse=True)[:3]
    print(f"\nQuery: {q}")
    for s, title in scores:
        print(f"  cos={s:.3f}  {title}")
