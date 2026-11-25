"""common.py - shared helpers for all CSA65 lab programs (A. ROHIT, 192472035).

* Hugging Face models are cached on E: (C: is nearly full on this laptop).
* gemini(prompt) - Google Gemini API (key from a .env file, never committed).
* local_llm(prompt) - a locally running LLM through Ollama (qwen3.5:2b).
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

from dotenv import load_dotenv  # noqa: E402
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "..", "CSA65_Proof")):
    load_dotenv(os.path.join(p, ".env"))

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")


def gemini(prompt, system=None, temperature=0.7):
    """Call Google Gemini with automatic retry / model fall-back. Returns text."""
    from google import genai
    from google.genai import types, errors
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing - create a .env file with GEMINI_API_KEY=your_key")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    last = None
    for m in GEMINI_MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=m, contents=prompt, config=cfg).text.strip()
            except errors.APIError as e:
                last = e
                if e.code not in (429, 500, 503, 504):
                    raise
                time.sleep(2 * (attempt + 1))
            except Exception as e:  # network hiccups (httpx disconnect / timeout) -> retry
                if not any(w in type(e).__name__ for w in ("Protocol", "Connect", "Timeout", "Read")):
                    raise
                last = e
                time.sleep(2 * (attempt + 1))
    raise last


def local_llm(prompt, system=None, temperature=0.3, model=None):
    """Call the locally running LLM through Ollama (server: `ollama serve`)."""
    import ollama
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = ollama.chat(model=model or OLLAMA_MODEL, messages=msgs, think=False,
                    options={"temperature": temperature, "num_predict": 400})
    return r["message"]["content"].strip()


def banner(title):
    print("=" * 70); print(title); print("A. ROHIT (192472035) - CSA65 Generative AI Lab"); print("=" * 70)


# ---------------- embeddings (Unit 3 / 4 / 5) ----------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_emb = None


def embedder():
    """Local sentence-embedding model (384-d) via fastembed, cached on E:."""
    global _emb
    if _emb is None:
        from fastembed import TextEmbedding
        _emb = TextEmbedding(EMBED_MODEL, cache_dir=os.path.join(os.environ["HF_HOME"], "fastembed"))
    return _emb


def embed(texts, query=False):
    """Return L2-normalised float32 numpy embeddings (rows) for a list of texts."""
    import numpy as np
    e = embedder()
    v = np.array(list(e.query_embed(texts) if query else e.passage_embed(texts)), dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_text(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size=400, overlap=80):
    """Sentence-aware chunking with character overlap."""
    import re
    sents, chunks, cur = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)), [], ""
    for s in sents:
        if len(cur) + len(s) > size and cur:
            chunks.append(cur.strip()); cur = cur[-overlap:] + " " + s
        else:
            cur += " " + s
    return chunks + ([cur.strip()] if cur.strip() else [])
