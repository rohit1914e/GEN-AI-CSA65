"""Unit 2 - P12: Prompt Evaluation, Comparison and Refinement using LLM Responses.
Several prompts for one task are scored on relevance, accuracy, completeness, clarity and format adherence
(LLM-as-judge with a fixed rubric + automatic checks); the best prompt is then refined and re-scored."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import json, re

banner("U2 P12 - Prompt evaluation and refinement")
TASK = "Explain what an API is for first-year engineering students."
prompts = {"P1 (bare)": "What is an API?",
           "P2 (audience)": "Explain what an API is to first-year engineering students.",
           "P3 (structured)": ("You are a lecturer. Explain what an API is to first-year engineering students in under 120 words, "
                               "with one real-life analogy and one Python example. Use 3 short bullet points.")}
RUBRIC = ("Score the RESPONSE for the TASK on relevance, accuracy, completeness, clarity and format_adherence (1-5 each). "
          'Return only JSON like {"relevance":5,"accuracy":5,"completeness":4,"clarity":5,"format_adherence":4,"comment":"..."}.')


def evaluate(prompt):
    resp = gemini(prompt, temperature=0.4)
    raw = gemini(f"{RUBRIC}\nTASK: {TASK}\nRESPONSE:\n{resp}", temperature=0)
    s = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    s["words"] = len(resp.split()); s["total"] = sum(s[k] for k in ("relevance", "accuracy", "completeness", "clarity", "format_adherence"))
    return resp, s


results = {}
for name, p in prompts.items():
    resp, s = evaluate(p); results[name] = (p, resp, s)
    print(f"{name:16s} total={s['total']:2d}/25  rel={s['relevance']} acc={s['accuracy']} comp={s['completeness']} "
          f"clar={s['clarity']} fmt={s['format_adherence']} words={s['words']} | {s['comment'][:70]}")
best = max(results, key=lambda k: results[k][2]["total"])
print("\nBest prompt:", best)
refined = gemini("Improve this prompt so the answer is clearer, more complete and strictly formatted. Return only the improved prompt.\n"
                 f"PROMPT: {results[best][0]}\nJUDGE FEEDBACK: {results[best][2]['comment']}", temperature=0.3)
resp, s = evaluate(refined)
print(f"\nRefined prompt:\n{refined}\n\nRefined total = {s['total']}/25 (was {results[best][2]['total']}/25)\n\nRefined response:\n{resp}")
