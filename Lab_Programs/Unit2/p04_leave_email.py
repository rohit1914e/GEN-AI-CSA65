"""Unit 2 - P4: Zero/One/Few-shot prompts for a professional leave email due to illness; compare tone, grammar, formatting, completeness."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
from shots import build  # noqa: E402

banner("U2 P4 - Sick-leave email: zero / one / few-shot")
task = "Input: Write a professional email to my manager requesting 2 days sick leave (fever) from 26 to 27 November 2026. Name: A. Rohit.\nOutput:"
examples = [("Email requesting 1 day leave for a family function", "Subject: Leave Request - 14 Oct 2026\n\nDear Ms. Priya,\n\nI would like to request leave on 14 October 2026 to attend a family function. I will complete my pending tasks beforehand and remain reachable by phone.\n\nKind regards,\nArun"),
            ("Email requesting half-day leave for a medical appointment", "Subject: Half-day Leave - 3 Nov 2026\n\nDear Mr. Kumar,\n\nI have a medical appointment on the afternoon of 3 November 2026 and request a half-day leave. My work will be handed over to Divya.\n\nThank you,\nMeena")]
outs = {n: gemini(p, temperature=0.5) for n, p in build(task, examples).items()}
for n, o in outs.items():
    print(f"\n--- {n.upper()} ---\n{o}")
print("\n=== Comparison ===\n" + gemini("Compare these three emails on tone, grammar, formatting and completeness (1-5 each) in a table and pick the best:\n" +
                                       "\n\n".join(f"{n.upper()}:\n{o}" for n, o in outs.items()), temperature=0))
