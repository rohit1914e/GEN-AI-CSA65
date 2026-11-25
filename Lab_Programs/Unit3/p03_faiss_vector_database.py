"""Unit 3 - P3: Build a vector database using FAISS and perform similarity-based document retrieval."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text, OUT  # noqa: E402
import faiss

banner("U3 P3 - FAISS vector database")
docs = [d for d in load_text("product_manuals.txt").split("\n\n") if d.strip()]
vecs = embed(docs)
index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
faiss.write_index(index, os.path.join(OUT, "U3_P03_manuals.faiss"))
print(f"Indexed {index.ntotal} documents, dimension {index.d}, saved to outputs/U3_P03_manuals.faiss")
for q in ["washing machine not draining water", "forgot wifi admin password", "laptop fan overheating"]:
    scores, ids = index.search(embed([q], query=True), 2)
    print(f"\nQuery: {q}")
    for s, i in zip(scores[0], ids[0]):
        print(f"  {s:.3f}  {docs[i].splitlines()[0]}")
