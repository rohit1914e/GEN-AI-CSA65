# CSA65 Generative AI – Lab Practice Programs

**A. ROHIT (192472035)** · All 62 programs from the "Lab Practice Programs" list (posted 26 Oct 2026).
`outputs/` holds the **real console output of every program** (`Unit3_p05_document_qa_rag.txt`, …) plus generated
images, audio, the FAISS index and the ChromaDB store.

| Unit | Programs | Main tools |
|---|---|---|
| **Unit1** (20) | Matplotlib, scikit-learn, NumPy, Pandas, PyTorch tensors, sentiment analysis, BERT tokenizer & contextual embeddings, GPT-2 generation, extractive QA, BERT vs GPT-2 tokens, similar-sentence embeddings | PyTorch, Hugging Face Transformers |
| **Unit2** (12) | Zero/one/few-shot prompts (product description, blog, 50-word summary, leave email, social post), OpenAI / Gemini / HF Inference APIs, Python code generation + automatic tests, SQL generation run on SQLite, measured comparison of prompting styles, prompt evaluation & refinement | Google Gemini API |
| **Unit3** (10) | Embeddings + similarity search, cosine search, FAISS vector DB, ChromaDB storage + top-k, RAG document QA, end-to-end RAG pipeline, LangChain domain chatbot, context-aware chatbot with history, assistant over external docs, multi-document (PDF+DOCX+TXT) assistant | fastembed (bge-small), FAISS, ChromaDB, LangChain, Gemini |
| **Unit4** (10) | College chatbot, TF-IDF engineering-support chatbot, text-to-image (bridge / robot), prompt-change image comparison, speech-to-text, text-to-speech, document summarizer, English→Hindi translation, resume screening & ranking, research assistant | Stable Diffusion Turbo, Whisper, Gemini TTS, MarianMT, scikit-learn |
| **Unit5** (10) | Streamlit text generation & summarization with a local LLM, local QA system, local translation/paraphrasing, Ollama text generation (streaming), Ollama QA, hallucination analysis, prompt-injection safeguards, local RAG, local RAG troubleshooting | Ollama (`qwen3.5:2b`), Streamlit, FAISS |

## Run
```bash
pip install -r requirements.txt
# .env (never committed):  GEMINI_API_KEY=...   optional: OPENAI_API_KEY=...  HF_TOKEN=...
python Unit1/p07_hf_sentiment_analysis.py            # any single program
streamlit run Unit5/p01_streamlit_text_generation.py # the two Streamlit apps
python run_all.py Unit1 Unit2 Unit3 Unit4 Unit5      # run everything and refresh outputs/
```
* Unit 5 needs Ollama running locally (`ollama serve`, model `qwen3.5:2b`).
* Hugging Face models are cached in `E:\hf_cache` (see `common.py`).
* Unit 2 P6 (OpenAI) and P8 (Hugging Face Inference API) need their own keys; without one they print a clear message instead of failing.
* Transformers v5 removed the question-answering pipeline, so Unit 1 P14 implements start/end-span prediction directly.
* Text-to-image uses Stable Diffusion Turbo locally because the free Gemini tier has no image-generation quota.

## Status
57 of 62 programs have been run and their outputs are in `outputs/`. Unit 4 P3, P4 (text-to-image), P8 (translation),
P9 (resume screening) and P10 (research assistant) are written but **not yet run** – their model downloads did not
finish – so they have no output files yet.
