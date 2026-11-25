"""Unit 2 - P3: Zero/One/Few-shot prompts to summarise an article into 50 words; compare accuracy, completeness, readability."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
from shots import build  # noqa: E402

banner("U2 P3 - 50-word summaries compared")
article = ("Electric vehicles (EVs) are growing quickly in India. Sales crossed two million units in 2025, led by two-wheelers "
           "and three-wheelers. Government subsidies under the FAME and PM E-DRIVE schemes reduced prices, while states added "
           "road-tax waivers. However, charging infrastructure is still limited outside big cities, and battery costs depend "
           "on imported lithium. Experts say local cell manufacturing and fast-charging corridors on highways are essential "
           "for EVs to reach 30 percent of new vehicle sales by 2030.")
task = f"Input: Summarise this article in exactly 50 words:\n{article}\nOutput:"
examples = [("Summarise in 50 words: (article on solar power growth)", "India's solar capacity reached 90 GW in 2025 thanks to falling panel prices and state incentives. Rooftop adoption is rising, but grid storage and land acquisition remain hurdles. Analysts expect domestic module manufacturing and battery storage tenders to drive the next phase of growth through 2030 and beyond.")]
outs = {n: gemini(p, temperature=0.3) for n, p in build(task, examples * 2).items()}
for n, o in outs.items():
    print(f"\n--- {n.upper()} ({len(o.split())} words) ---\n{o}")
judge = gemini("Rate each summary 1-5 for accuracy, completeness and readability against the article, as a compact table, then name the best.\n"
               f"ARTICLE: {article}\n" + "\n".join(f"{n.upper()}: {o}" for n, o in outs.items()), temperature=0)
print("\n=== Comparison (LLM-as-judge) ===\n" + judge)
