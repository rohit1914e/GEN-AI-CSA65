"""Unit 1 - P14: Question Answering using a pre-trained Transformer.

Extractive QA with DistilBERT fine-tuned on SQuAD. (transformers v5 removed the
"question-answering" pipeline, so the start/end-span logic is written explicitly.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

banner("P14 - Extractive question answering (DistilBERT fine-tuned on SQuAD)")
name = "distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForQuestionAnswering.from_pretrained(name)
context = ("Saveetha School of Engineering is located in Chennai, Tamil Nadu. The CSA65 course covers transformers, "
           "prompt engineering, retrieval-augmented generation and AI agents. Students need 75 percent attendance.")


def answer(question):
    enc = tok(question, context, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    start = int(out.start_logits.argmax())
    end = int(out.end_logits[0, start:].argmax()) + start          # end must not come before start
    score = (out.start_logits.softmax(-1)[0, start] * out.end_logits.softmax(-1)[0, end]).item()
    return tok.decode(enc["input_ids"][0, start:end + 1]), score


print("Context:", context, "\n")
for q in ["Where is Saveetha School of Engineering located?", "What does CSA65 cover?", "How much attendance is needed?"]:
    a, s = answer(q)
    print(f"Q: {q}\nA: {a}  (confidence {s:.3f})\n")
