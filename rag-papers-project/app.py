import streamlit as st
from generate_answer import generate_answer

st.set_page_config(page_title="RAG over RAG Papers", page_icon="📄")

st.title("📄 Ask Questions About RAG Research")
st.write(
    "This is a Retrieval-Augmented Generation (RAG) system built to answer "
    "questions about 6 research papers on RAG itself — using only the text "
    "of those papers, not the model's general knowledge."
)

with st.expander("📚 Papers in this knowledge base"):
    st.markdown("""
    1. Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    2. Izacard & Grave (2020) — Fusion-in-Decoder
    3. Gao et al. (2023) — RAG for LLMs: A Survey
    4. Asai et al. (2023) — Self-RAG
    5. Es et al. (2023) — RAGAS: Automated Evaluation of RAG
    6. Recent 2025 RAG Survey
    """)

question = st.text_input("Ask a question about RAG:", placeholder="e.g. What is Self-RAG and how does it differ from naive RAG?")

if question:
    with st.spinner("Retrieving relevant passages and generating answer..."):
        answer, sources = generate_answer(question)
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### Sources used")
    for s in set(sources):
        st.write(f"- {s}")