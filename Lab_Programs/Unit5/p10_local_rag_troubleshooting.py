"""Unit 5 - P10: Local RAG-based engineering troubleshooting system that retrieves relevant information from local technical
documents and generates step-by-step recommendations using Ollama."""
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

banner("U5 P10 - Local RAG troubleshooting assistant (step-by-step)")
index, chunks = build(["engineering_troubleshooting.txt", "product_manuals.txt"], size=500)
for problem in ["The hydraulic press is noisy and not building pressure", "Our CNC lathe parts have a rough surface finish",
                "Robot welds are full of tiny holes"]:
    hits = retrieve(index, chunks, problem, k=2)
    ctx = "\n".join(c for _, _, c in hits)
    steps = local_llm("You are a maintenance engineer. Using ONLY the manual extract, give numbered step-by-step troubleshooting "
                      f"actions (max 5 steps) and a safety note.\nMANUAL:\n{ctx}\nPROBLEM: {problem}", temperature=0)
    print(f"\nProblem  : {problem}\nRetrieved: {[c[:45] + '...' for _, _, c in hits]}\n{steps}")
