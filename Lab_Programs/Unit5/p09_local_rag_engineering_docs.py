"""Unit 5 - P9: Local Retrieval-Augmented Generation system using engineering documents, a vector database (FAISS)
and Ollama to answer technical questions - everything runs offline on this laptop."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402
import faiss
from common import embed, chunk_text, load_text, local_llm


def build(files, size=350):
    chunks = [(f, c) for f in files for c in chunk_text(load_text(f), size, 60)]
    index = faiss.IndexFlatIP(384); index.add(embed([c for _, c in chunks]))
    return index, chunks


def retrieve(index, chunks, q, k=3):
    s, ids = index.search(embed([q], query=True), k)
    return [(float(x), *chunks[i]) for x, i in zip(s[0], ids[0])]

banner("U5 P9 - Local RAG: engineering notes + FAISS + Ollama")
index, chunks = build(["os_memory_notes.txt", "computer_networks_notes.txt", "dbms_normalization_notes.txt", "product_manuals.txt"])
print("Indexed chunks:", len(chunks))
for q in ["What is the TLB and why is it used?", "How does TCP slow start work?", "How do I clean the washing machine drum?",
          "What is the boiling point of mercury?"]:
    hits = retrieve(index, chunks, q)
    if hits[0][0] < 0.55:
        print(f"\nQ: {q}\nA: Not covered by the local documents (best similarity {hits[0][0]:.2f})."); continue
    ctx = "\n".join(f"[{f}] {c}" for _, f, c in hits)
    a = local_llm(f"Answer the question using only the context. Mention the source file.\nCONTEXT:\n{ctx}\nQUESTION: {q}", temperature=0)
    print(f"\nQ: {q}\nRetrieved: {[(f, round(s, 2)) for s, f, _ in hits]}\nA: {a}")
