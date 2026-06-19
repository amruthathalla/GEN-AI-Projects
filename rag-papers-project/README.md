# RAG over RAG Papers

A Retrieval-Augmented Generation system that answers questions about 6 foundational research papers on RAG itself — grounded in the actual paper text, with source citations, rather than relying on the LLM's general knowledge.

## Why this project

I wanted to understand RAG by building one, not just reading about it. Using the system to answer questions about the research that defines RAG felt like a fitting way to test it.

## How it works

1. **Ingest**: 6 PDFs are parsed into raw text (`pypdf`).
2. **Chunk**: Text is split into 500-word chunks with 50-word overlap, to keep retrieval units small and meaningful while avoiding hard cuts mid-idea.
3. **Embed**: Each chunk is converted into a vector embedding (`sentence-transformers`, `all-MiniLM-L6-v2`) — a free, local model, no API cost.
4. **Store**: Embeddings are stored in a local vector database (ChromaDB).
5. **Retrieve**: A user's question is embedded the same way, and the database returns the most semantically similar chunks.
6. **Generate**: Retrieved chunks + the question are passed to an LLM (Llama 3.1 8B via Groq's free API), explicitly instructed to answer only from the provided context and to admit when it can't.
7. **Interface**: A Streamlit app exposes this as a simple Q&A tool with source attribution.

## Design decisions and tradeoffs

- **Chunk size (500 words, 50 overlap)**: balances having enough context per chunk against retrieval precision. Smaller chunks risk losing context; larger chunks risk diluting relevance.
- **Retrieving top 4 chunks**: a tradeoff between giving the LLM enough evidence and avoiding noisy, irrelevant context.
- **Explicit "don't guess" instruction**: reduces hallucination by constraining the model to the retrieved evidence, and tested this directly with out-of-scope questions.

## What I'd improve with more time

- Add a re-ranking step after retrieval to improve precision on ambiguous questions.
- Build a small evaluation set (e.g. 20 question/answer pairs) and score retrieval/answer quality systematically, using ideas from the RAGAS paper itself.
- Experiment with chunk size empirically rather than choosing it by convention.

## Stack

Python, sentence-transformers, ChromaDB, Groq API (Llama 3.1), Streamlit

## Papers used


* [https://arxiv.org/pdf/2005.11401](https://arxiv.org/pdf/2005.11401) — Lewis et al., original RAG paper
* [https://arxiv.org/pdf/2007.01282](https://arxiv.org/pdf/2007.01282) — Izacard & Grave, Fusion-in-Decoder
* [https://arxiv.org/pdf/2312.10997](https://arxiv.org/pdf/2312.10997) — Gao et al., RAG survey
* [https://arxiv.org/pdf/2310.11511](https://arxiv.org/pdf/2310.11511) — Asai et al., Self-RAG
* [https://arxiv.org/pdf/2309.15217](https://arxiv.org/pdf/2309.15217) — Es et al., RAGAS
* [https://arxiv.org/pdf/2506.00054](https://arxiv.org/pdf/2506.00054) — recent RAG survey (2025)
