"""Unit 4 - P2: Engineering-support chatbot that answers technical questions with classic NLP techniques
(tokenisation, stop-word removal, lemma-like stemming, TF-IDF vectors, cosine similarity) - no LLM needed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, load_text  # noqa: E402
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

banner("U4 P2 - Engineering-support chatbot (TF-IDF NLP)")
entries = [e.split("\n", 1) for e in load_text("engineering_troubleshooting.txt").strip().split("\n\n")]


def normalise(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stem = lambda w: re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w
    return " ".join(stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS)


vec = TfidfVectorizer(ngram_range=(1, 2))
M = vec.fit_transform([normalise(t + " " + b) for t, b in entries])
print("Vocabulary size:", len(vec.vocabulary_), "| knowledge entries:", len(entries))
for q in ["my motor is getting too hot", "welds have small holes / porosity", "PLC not talking to the HMI screen",
          "compressor keeps tripping", "how do I bake a cake"]:
    sims = cosine_similarity(vec.transform([normalise(q)]), M)[0]
    i = sims.argmax()
    print(f"\nUser : {q}\nTokens: {normalise(q)}")
    print(f"Bot  : [{entries[i][0]}] (score {sims[i]:.2f})\n       {entries[i][1]}" if sims[i] > 0.1 else
          f"Bot  : Sorry, I have no solution for that (best score {sims[i]:.2f}).")
