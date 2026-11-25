"""Unit 5 - P4: Integrate a locally running LLM with Python to perform text translation and paraphrasing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, local_llm  # noqa: E402

banner("U5 P4 - Translation and paraphrasing with a local LLM")
text = "Regular maintenance of machines reduces breakdowns and improves workplace safety."
print("Original:", text, "\n")
for lang in ["French", "Hindi", "Tamil"]:
    out = local_llm(f"Translate into {lang}. Output only the translation.\n{text}", temperature=0)
    print(f"{lang:8s}: {out}")
print()
for style in ["simpler words for a school student", "formal business tone", "a single short sentence of max 8 words"]:
    out = local_llm(f"Paraphrase in {style}. Output only the sentence.\n{text}", temperature=0.4)
    print(f"Paraphrase ({style}): {out}")
