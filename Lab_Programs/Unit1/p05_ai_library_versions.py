"""Unit 1 - P5: Verify the installed versions of all AI libraries on your system."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import importlib
import platform

banner("P5 - Installed AI library versions")
print("Python", platform.python_version(), "on", platform.system(), platform.release())
libs = ["numpy", "pandas", "matplotlib", "sklearn", "torch", "transformers", "tokenizers", "sentencepiece",
        "faiss", "chromadb", "fastembed", "langchain", "google.genai", "ollama", "faster_whisper", "streamlit"]
for name in libs:
    try:
        m = importlib.import_module(name)
        print(f"  {name:15s} {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"  {name:15s} NOT INSTALLED ({type(e).__name__})")
