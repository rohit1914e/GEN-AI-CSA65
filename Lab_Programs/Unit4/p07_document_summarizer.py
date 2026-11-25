"""Unit 4 - P7: Summarise a lengthy engineering document into a short, meaningful summary using a pre-trained language model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, load_text, gemini  # noqa: E402

banner("U4 P7 - Engineering document summarizer")
doc = load_text("engineering_troubleshooting.txt") + "\n\n" + load_text("product_manuals.txt")
summary = gemini("Summarise this engineering document in 5 bullet points (max 80 words total), keeping key numbers.\n\n" + doc,
                 temperature=0.2)
print(f"Original: {len(doc.split())} words\nSummary : {len(summary.split())} words "
      f"(compression {len(summary.split()) / len(doc.split()):.0%})\n\n{summary}")
