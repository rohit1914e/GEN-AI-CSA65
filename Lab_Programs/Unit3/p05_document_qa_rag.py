"""Unit 3 - P5: Develop a document Question Answering system using Retrieval-Augmented Generation (RAG)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, load_text, chunk_text, gemini  # noqa: E402
import faiss

banner("U3 P5 - Document QA with RAG (FAISS + Gemini)")
chunks = chunk_text(load_text("os_memory_notes.txt"), 350, 70)
index = faiss.IndexFlatIP(384); index.add(embed(chunks))


def ask(q, k=3, threshold=0.55):
    s, ids = index.search(embed([q], query=True), k)
    if s[0][0] < threshold:
        return "Not available in the document.", s[0][0]
    ctx = "\n".join(f"[{n + 1}] {chunks[i]}" for n, i in enumerate(ids[0]))
    return gemini(f"Answer only from the context and cite [n].\nContext:\n{ctx}\nQuestion: {q}", temperature=0.1), s[0][0]


for q in ["What is a page fault?", "What is Belady's anomaly?", "Who invented the internet?"]:
    a, best = ask(q)
    print(f"\nQ: {q}  (best similarity {best:.2f})\nA: {a}")
