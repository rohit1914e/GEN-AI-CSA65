# CSA65 Generative AI – Lab Practice Programs

**A. ROHIT (192472035)** · Programs from the "Lab Practice Programs" list (posted 26 Oct 2026).
This folder currently covers **30 programs: Unit 1 (P1–P20) and Unit 2 (P1–P10)**.

| Unit | Programs | Main tools |
|---|---|---|
| Unit1 | p01–p20: Matplotlib, scikit-learn, NumPy, Pandas, PyTorch tensors, HF sentiment analysis, BERT tokenizer & contextual embeddings, GPT-2 generation, extractive QA, BERT vs GPT-2 tokens | PyTorch, Hugging Face Transformers |
| Unit2 | p01–p05 zero/one/few-shot prompting (product description, blog, 50-word summary, leave email, social post), p06 OpenAI API, p07 Gemini API, p08 HF Inference API, p09 Python code generation + automatic validation, p10 SQL generation run on SQLite | Google Gemini API |

`outputs/` holds the real console output of every program (`Unit1_p07_...txt` etc.) and `U1_P01_line_graph.png`.

## Run
```bash
pip install -r requirements.txt
# .env (NOT committed):  GEMINI_API_KEY=...   optional: OPENAI_API_KEY=...  HF_TOKEN=...
python Unit1/p07_hf_sentiment_analysis.py      # any single program
python run_all.py Unit1 Unit2                  # run everything and refresh outputs/
```
Notes: Hugging Face models are cached in `E:\hf_cache` (see `common.py`). Unit 2 P6 (OpenAI) and P8 (Hugging Face
Inference API) need their own keys; without a key they print a clear message instead of failing (see their outputs).
Transformers v5 removed the question-answering pipeline, so Unit 1 P14 implements start/end span prediction directly.
