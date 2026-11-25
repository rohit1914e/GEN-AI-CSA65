"""Unit 5 - P7: Demonstrate and analyse hallucinations in a locally running LLM using suitable prompts and reference information.
The model is asked about (a) facts only in a private reference and (b) a fictional entity, first WITHOUT and then WITH the
reference; answers are checked against the reference automatically."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, local_llm  # noqa: E402

banner("U5 P7 - Hallucination demo and analysis (local LLM)")
REFERENCE = ("NovaBook 14 laptop: 65 W USB-C charger, battery life up to 12 hours in Battery Saver mode, "
             "Fn + F5 toggles the keyboard backlight, the 80% charge limit is enabled in the NovaCare app.")
tests = [("What wattage is the NovaBook 14 charger?", "65"),
         ("Which key combination toggles the NovaBook 14 keyboard backlight?", "f5"),
         ("Who won the 2026 Nobel Prize for the NovaBook 14 laptop design?", None)]       # trick question
for q, truth in tests:
    plain = local_llm(q + " Answer in one sentence.", temperature=0.7)
    grounded = local_llm(f"Use ONLY this reference; if the answer is not in it, say 'Not in the reference'.\n"
                         f"REFERENCE: {REFERENCE}\nQUESTION: {q}", temperature=0)
    check = lambda a: ("correct" if truth and truth in a.lower() else "refused (good)" if not truth and "not" in a.lower()
                       else "HALLUCINATION / unsupported")
    print(f"\nQ: {q}\n  Without reference: {plain}\n    -> {check(plain)}\n  With reference   : {grounded}\n    -> {check(grounded)}")
print("\nAnalysis: without grounding the small local model invents specific-sounding facts; giving reference text and "
      "an explicit 'say not in reference' rule removes most hallucinations.")
