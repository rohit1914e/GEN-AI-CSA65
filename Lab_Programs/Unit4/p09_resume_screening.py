"""Unit 4 - P9: AI-based Resume Screening - analyse candidate resumes and rank them against an engineering job description.
Score = 0.6 x semantic similarity (embeddings) + 0.4 x required-skill coverage."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, embed  # noqa: E402
import re

banner("U4 P9 - Resume screening and ranking")
JD = ("Machine Learning Engineer: Python, PyTorch, NLP, transformers, SQL, Docker. Build and deploy LLM-based applications "
      "and RAG pipelines. 1-3 years experience.")
SKILLS = ["python", "pytorch", "nlp", "transformers", "sql", "docker", "rag", "llm"]
resumes = {
    "Ananya": "B.E. CSE. 2 years at a startup building RAG chatbots with Python, LangChain, transformers and FAISS. Docker, SQL.",
    "Vikram": "Mechanical engineer, AutoCAD, SolidWorks, 3 years in automotive design, basic Python scripting.",
    "Sneha": "Data analyst: SQL, Excel, Power BI, Python pandas. Some exposure to NLP sentiment analysis.",
    "Rahul": "ML intern: PyTorch CNN image classification, fine-tuned BERT for NLP text classification, deployed LLM API with Docker.",
    "Meera": "Web developer: React, Node.js, MongoDB, HTML/CSS, 4 years experience.",
}
jd_vec = embed([JD], query=True)[0]
rows = []
for name, text in resumes.items():
    sim = float(embed([text])[0] @ jd_vec)
    have = [s for s in SKILLS if re.search(rf"\b{s}\b", text.lower())]
    rows.append((0.6 * sim + 0.4 * len(have) / len(SKILLS), sim, name, have))
print(f"{'Rank':4s} {'Candidate':9s} {'Score':>6s} {'Semantic':>9s}  Matched skills")
for r, (score, sim, name, have) in enumerate(sorted(rows, reverse=True), 1):
    print(f"{r:<4d} {name:9s} {score:6.3f} {sim:9.3f}  {', '.join(have) or '-'}")
