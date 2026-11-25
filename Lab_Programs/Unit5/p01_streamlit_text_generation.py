"""Unit 5 - P1: Streamlit application for text generation using a locally running LLM (Ollama, qwen3.5:2b).
Run the app:  streamlit run Unit5/p01_streamlit_text_generation.py      (plain `python` runs a console self-test)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, local_llm, OLLAMA_MODEL  # noqa: E402

def in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


def generate(prompt, style, words):
    return local_llm(f"Write a {style} text of about {words} words: {prompt}", temperature=0.7)


if in_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Local LLM Text Generator", page_icon="✍️")
    st.title("✍️ Text Generation with a Local LLM")
    st.caption(f"A. ROHIT (192472035) · Ollama model `{OLLAMA_MODEL}` running on this laptop - no internet needed")
    prompt = st.text_area("Prompt", "A short story about a robot that learns to paint")
    c1, c2 = st.columns(2)
    style = c1.selectbox("Style", ["creative", "formal", "technical", "funny"])
    words = c2.slider("Length (words)", 30, 200, 80)
    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Local model is writing..."):
                st.success(generate(prompt, style, words))
else:
    banner("U5 P1 - Streamlit text generation (console self-test)")
    print(generate("A short story about a robot that learns to paint", "creative", 60))
