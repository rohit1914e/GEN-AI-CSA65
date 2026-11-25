"""Unit 2 - P1: Zero-shot, One-shot and Few-shot prompts to generate a product description for a Smart Fitness Watch."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
from shots import build  # noqa: E402

banner("U2 P1 - Product description: zero / one / few-shot")
task = "Input: Smart Fitness Watch (heart-rate, SpO2, GPS, 10-day battery, Rs. 4,999)\nOutput:"
examples = [("Wireless Earbuds (ANC, 30h battery, Rs. 2,499)", "Silence the world. 30 hours of music, active noise cancellation - all for Rs. 2,499."),
            ("Power Bank (20000 mAh, 22.5W fast charge)", "Charge your phone four times over. 20000 mAh with 22.5W fast charging.")]
for name, p in build(task, examples).items():
    out = gemini(p, temperature=0.7)
    print(f"\n--- {name.upper()} ({len(out.split())} words) ---\n{out}")
