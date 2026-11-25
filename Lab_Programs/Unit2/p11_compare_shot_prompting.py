"""Unit 2 - P11: Comparative Analysis of Zero-shot, One-shot and Few-shot Prompting.
Same task (review sentiment + aspect extraction) with three prompting styles; quality, relevance, accuracy and
consistency are measured objectively against labelled answers, each prompt run 3 times for consistency."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import json, re

banner("U2 P11 - Zero vs One vs Few-shot: measured comparison")
reviews = [("Battery lasts two days but the camera is blurry at night.", "mixed"),
           ("Delivery was late and the box was damaged.", "negative"),
           ("Super fast laptop, great keyboard, worth every rupee!", "positive"),
           ("The fan is noisy, otherwise it works okay.", "mixed"),
           ("Terrible app, crashes every time I log in.", "negative"),
           ("Loved the food and the staff were very polite.", "positive")]
task = 'Classify the review as positive, negative or mixed. Reply ONLY as JSON {"sentiment": "..."}.\nReview: '
ex = [("Screen is bright but speakers are weak.", "mixed"), ("Excellent build quality!", "positive"), ("Stopped working in a week.", "negative")]
shots = {"zero-shot": "", "one-shot": f"Review: {ex[0][0]}\nAnswer: {{\"sentiment\": \"{ex[0][1]}\"}}\n\n",
         "few-shot": "".join(f"Review: {a}\nAnswer: {{\"sentiment\": \"{b}\"}}\n\n" for a, b in ex)}
print(f"{'Style':10s} {'Accuracy':>9s} {'Format OK':>10s} {'Consistency':>12s}")
for style, prefix in shots.items():
    correct = fmt = consistent = 0
    for text, label in reviews:
        answers = []
        for _ in range(3):
            raw = gemini(prefix + task + text, temperature=0.9)
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                answers.append(json.loads(m.group(0))["sentiment"].lower()); fmt += 1
            except Exception:
                answers.append("invalid")
        correct += answers[0] == label
        consistent += len(set(answers)) == 1
    n = len(reviews)
    print(f"{style:10s} {correct / n:9.0%} {fmt / (3 * n):10.0%} {consistent / n:12.0%}")
print("\nAccuracy = correct label (first run); Format OK = valid JSON; Consistency = same answer in all 3 runs.")
