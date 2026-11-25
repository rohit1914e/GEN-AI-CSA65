"""Unit 2 - P2: Zero/One/Few-shot prompts to generate a 200-word blog on 'Applications of Artificial Intelligence in Healthcare'."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
from shots import build  # noqa: E402

banner("U2 P2 - 200-word blog: zero / one / few-shot")
task = "Input: Write a 200-word blog on 'Applications of Artificial Intelligence in Healthcare'.\nOutput:"
examples = [("Write a 200-word blog on 'AI in Agriculture'", "Title: Smarter Farms with AI\nIntro... (hook sentence)\n1. Crop disease detection...\n2. Precision irrigation...\n3. Yield prediction...\nConclusion: ... (call to action)"),
            ("Write a 200-word blog on 'AI in Education'", "Title: Classrooms that Learn\nIntro...\n1. Personalised learning...\n2. Automated grading...\n3. Early warning systems...\nConclusion: ...")]
for name, p in build(task, examples).items():
    out = gemini(p)
    print(f"\n--- {name.upper()} : {len(out.split())} words (target 200) ---\n{out}")
