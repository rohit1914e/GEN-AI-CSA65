"""Unit 5 - P5: Demonstrate text generation using a local Large Language Model through Ollama and Python
(ollama Python client, streaming tokens, model info, generation speed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OLLAMA_MODEL  # noqa: E402
import time
import ollama

banner("U5 P5 - Text generation through Ollama")
info = ollama.show(OLLAMA_MODEL)
print("Model:", OLLAMA_MODEL, "| family:", info["details"]["family"], "| parameters:", info["details"]["parameter_size"],
      "| quantization:", info["details"]["quantization_level"])
prompt = "Write a 4-line poem about engineers building bridges."
print("Prompt:", prompt, "\nOutput (streamed): ", end="", flush=True)
t, n = time.time(), 0
for part in ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=True, think=False, options={"temperature": 0.8}):
    print(part["response"], end="", flush=True); n += 1
    if part.get("done"):
        print(f"\n\n{part['eval_count']} tokens in {part['eval_duration'] / 1e9:.1f}s "
              f"= {part['eval_count'] / (part['eval_duration'] / 1e9):.1f} tokens/s (total {time.time() - t:.1f}s)")
