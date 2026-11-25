"""Unit 2 - P5: Zero/One/Few-shot prompts to generate a promotional social media post for an AI Workshop."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
from shots import build  # noqa: E402

banner("U2 P5 - Social media post: zero / one / few-shot")
task = "Input: Promotional Instagram post for a 2-day Generative AI Workshop at SIMATS, 5-6 Dec 2026, free for students, register by 1 Dec.\nOutput:"
examples = [("Post for a Python bootcamp", "Code your future! 3-day Python Bootcamp, 10-12 Jan. Hands-on projects + certificate. Seats limited - register now! #Python #Coding #Bootcamp"),
            ("Post for a cybersecurity seminar", "Can you spot a phishing link? Join our Cybersecurity Seminar, 20 Feb, Main Hall. Free entry! #CyberSecurity #StaySafe")]
for name, p in build(task, examples).items():
    out = gemini(p, temperature=0.8)
    print(f"\n--- {name.upper()} ({len(out)} chars, {out.count('#')} hashtags) ---\n{out}")
