"""Unit 4 - P10: AI-based Research Assistant - accepts a research topic and generates relevant information, keywords and a concise summary."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import json, re

banner("U4 P10 - Research assistant")
topic = " ".join(sys.argv[1:]) or "Retrieval-Augmented Generation for reducing LLM hallucinations"
raw = gemini("You are a research assistant. For the topic below return ONLY JSON with keys: overview (3 sentences), "
             "key_concepts (list of 5), keywords (list of 8), research_questions (list of 3), summary (max 40 words).\n"
             f"TOPIC: {topic}", temperature=0.3)
r = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
print("Topic   :", topic)
print("\nOverview:", r["overview"])
print("\nKey concepts:"); [print("  -", c) for c in r["key_concepts"]]
print("\nKeywords:", ", ".join(r["keywords"]))
print("\nOpen research questions:"); [print("  ?", q) for q in r["research_questions"]]
print("\nSummary :", r["summary"])
