"""Unit 3 - P9: Simple AI assistant capable of answering questions using external documents (all .txt files in data/)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed, chunk_text, gemini, DATA  # noqa: E402
import glob
import faiss

banner("U3 P9 - AI assistant over external documents")
chunks = []
for p in glob.glob(os.path.join(DATA, "*.txt")):
    chunks += [(os.path.basename(p), c) for c in chunk_text(open(p, encoding="utf-8").read(), 400, 60)]
index = faiss.IndexFlatIP(384); index.add(embed([c for _, c in chunks]))
print(f"Knowledge base: {len(set(s for s, _ in chunks))} documents, {len(chunks)} chunks")


def assistant(q):
    s, ids = index.search(embed([q], query=True), 3)
    if s[0][0] < 0.55:
        return "I could not find this in the documents.", []
    hits = [chunks[i] for i in ids[0]]
    ctx = "\n".join(f"[{src}] {txt}" for src, txt in hits)
    return gemini(f"Answer using only the documents and cite [file].\n{ctx}\nQuestion: {q}", temperature=0.1), sorted({h[0] for h in hits})


for q in ["What should I do if the hydraulic press has low pressure?", "What is the hostel fee?", "How do I fix router internet LED orange?", "Who is the Prime Minister?"]:
    a, src = assistant(q)
    print(f"\nQ: {q}\nSources: {src}\nA: {a}")
