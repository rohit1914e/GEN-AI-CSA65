"""Unit 1 - P7: Load a pre-trained Transformer model using Hugging Face and perform sentiment analysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

banner("P7 - Sentiment analysis with a pre-trained Transformer")
name = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
texts = ["The Generative AI lab session was really interesting!", "The exam hall was too hot and noisy."]
with torch.no_grad():
    logits = model(**tok(texts, return_tensors="pt", padding=True)).logits
for t, p in zip(texts, logits.softmax(-1)):
    label = model.config.id2label[int(p.argmax())]
    print(f"{label:8s} ({p.max():.3f})  {t}")
