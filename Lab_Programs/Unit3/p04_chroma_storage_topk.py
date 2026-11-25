"""Unit 3 - P4: Implement document storage and top-k retrieval using a vector database (ChromaDB, persistent)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text, OUT  # noqa: E402
import chromadb

banner("U3 P4 - Document storage + top-k retrieval with ChromaDB")
client = chromadb.PersistentClient(path=os.path.join(OUT, "U3_P04_chroma_db"))
col = client.get_or_create_collection("handbook", metadata={"hnsw:space": "cosine"})
sections = [s for s in load_text("student_handbook.txt").split("\n\n") if s.strip()]
col.upsert(ids=[f"H{i}" for i in range(len(sections))], documents=sections, embeddings=embed(sections).tolist(),
           metadatas=[{"title": s.splitlines()[0], "source": "student_handbook.txt"} for s in sections])
print("Stored documents:", col.count(), "(persistent at outputs/U3_P04_chroma_db)")
for q, k in [("What is the pass mark?", 1), ("rules inside the laboratory", 3), ("can I bring my phone to the exam", 2)]:
    r = col.query(query_embeddings=embed([q], query=True).tolist(), n_results=k)
    print(f"\nQuery: {q}  (top-{k})")
    for doc_id, meta, dist in zip(r["ids"][0], r["metadatas"][0], r["distances"][0]):
        print(f"  {doc_id}  similarity={1 - dist:.3f}  {meta['title']}")
