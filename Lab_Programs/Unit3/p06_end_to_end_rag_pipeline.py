"""Unit 3 - P6: End-to-end RAG pipeline: document loading -> text chunking -> embeddings -> retrieval -> answer generation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, chunk_text, gemini, DATA  # noqa: E402
import glob
import faiss

banner("U3 P6 - End-to-end RAG pipeline")
# 1. Load
docs = {os.path.basename(p): open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(DATA, "*_notes.txt"))}
print("1. Loaded:", {k: len(v) for k, v in docs.items()}, "characters")
# 2. Chunk
chunks = [(name, c) for name, t in docs.items() for c in chunk_text(t, 400, 80)]
print("2. Chunks:", len(chunks), "(400 chars, 80 overlap)")
# 3. Embed + 4. Index
vecs = embed([c for _, c in chunks]); index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
print("3. Embeddings:", vecs.shape, "-> 4. FAISS index with", index.ntotal, "vectors")
# 5. Retrieve + 6. Generate
for q in ["How does TCP avoid congestion?", "When is a table in BCNF?"]:
    s, ids = index.search(embed([q], query=True), 3)
    ctx = "\n".join(f"({chunks[i][0]}) {chunks[i][1]}" for i in ids[0])
    print(f"\n5. Retrieved for '{q}':", [(chunks[i][0], round(float(x), 2)) for x, i in zip(s[0], ids[0])])
    print("6. Answer:", gemini(f"Using only this context, answer briefly and name the source file.\n{ctx}\nQ: {q}", temperature=0.1))
