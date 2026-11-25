"""Unit 5 - P2: Streamlit application for text summarization using a locally running LLM (Ollama).
Run the app:  streamlit run Unit5/p02_streamlit_summarization.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, local_llm, load_text, OLLAMA_MODEL  # noqa: E402

def in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


def summarize(text, n):
    return local_llm(f"Summarize the following text in {n} bullet points. Keep key numbers.\n\n{text}", temperature=0.2)


SAMPLE = load_text("engineering_troubleshooting.txt")[:1400]
if in_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Local LLM Summarizer", page_icon="📝")
    st.title("📝 Text Summarization with a Local LLM")
    st.caption(f"A. ROHIT (192472035) · Ollama `{OLLAMA_MODEL}`")
    text = st.text_area("Text to summarize", SAMPLE, height=220)
    n = st.slider("Bullet points", 2, 6, 3)
    if st.button("Summarize", type="primary"):
        if len(text.split()) < 20:
            st.warning("Please paste at least 20 words.")
        else:
            with st.spinner("Summarizing locally..."):
                s = summarize(text, n)
            st.markdown(s); st.caption(f"{len(text.split())} words -> {len(s.split())} words")
else:
    banner("U5 P2 - Streamlit summarization (console self-test)")
    s = summarize(SAMPLE, 3); print(s); print(f"\n{len(SAMPLE.split())} words -> {len(s.split())} words")
