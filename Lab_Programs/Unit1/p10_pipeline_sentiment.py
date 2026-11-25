"""Unit 1 - P10: Load a pre-trained transformer model using the Hugging Face pipeline() API and perform sentiment analysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import pipeline

banner("P10 - Sentiment analysis with pipeline()")
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
reviews = ["I love how easy this phone is to use.", "Worst customer service I have ever seen.",
           "The movie was okay, not great but not bad.", "Absolutely brilliant explanation by the professor!"]
for r, res in zip(reviews, clf(reviews)):
    print(f"{res['label']:8s} {res['score']:.3f} | {r}")
