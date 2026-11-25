"""Unit 5 - P6: Demonstrate question answering using a local Large Language Model through Ollama and Python (chat API)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OLLAMA_MODEL  # noqa: E402
import ollama

banner("U5 P6 - Question answering through Ollama")
messages = [{"role": "system", "content": "You are a concise engineering tutor. Answer in at most 2 sentences."}]
for q in ["What is Ohm's law?", "What is the difference between stress and strain?", "Give an example where Ohm's law is used."]:
    messages.append({"role": "user", "content": q})
    a = ollama.chat(model=OLLAMA_MODEL, messages=messages, think=False, options={"temperature": 0})["message"]["content"].strip()
    messages.append({"role": "assistant", "content": a})
    print(f"\nQ: {q}\nA: {a}")
