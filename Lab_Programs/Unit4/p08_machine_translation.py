"""Unit 4 - P8: Machine translation of an engineering document from English into an Indian language using a pre-trained
translation model (Helsinki-NLP MarianMT opus-mt-en-hi, English -> Hindi)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402
from transformers import MarianMTModel, MarianTokenizer

banner("U4 P8 - English -> Hindi translation (MarianMT)")
name = "Helsinki-NLP/opus-mt-en-hi"
tok = MarianTokenizer.from_pretrained(name); model = MarianMTModel.from_pretrained(name)
sentences = ["Check the oil level before starting the hydraulic press.",
             "The motor overheats when the cooling vents are blocked.",
             "Always wear safety goggles in the workshop."]
out = model.generate(**tok(sentences, return_tensors="pt", padding=True), max_new_tokens=80)
for en, hi in zip(sentences, tok.batch_decode(out, skip_special_tokens=True)):
    print(f"EN: {en}\nHI: {hi}\n")
