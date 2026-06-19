# Hi, I'm Amrutha Varshini Thalla

I build with Generative AI — specifically, systems where a language model's output is grounded in retrieved knowledge or shaped through fine-tuning, rather than left to the model's own defaults. I'm interested in retrieval and information access in particular: how a model decides what knowledge to rely on, and how that decision shapes everything it produces afterward. My background is in Information Science (M.S., Indiana Institute of Technology, USA, May 2025), which gave me an entry point into this through the field's older questions — how knowledge is organized, indexed, and made findable — rather than through model architecture first. That framing is still how I approach retrieval-augmented generation today.

I learn GenAI by building it end-to-end: chunking and embedding strategies, vector retrieval, prompting for grounded generation, and parameter-efficient fine-tuning.

## GenAI projects

- **RAG over RAG research papers** — a retrieval-augmented Q&A system grounded in six foundational RAG papers themselves, with source citations and an explicit instruction to admit uncertainty rather than guess. Covers the full pipeline: chunking, embedding (`sentence-transformers`), vector storage (ChromaDB), retrieval, and grounded generation (Llama 3.1 via Groq).
- **PDF chatbot (RAG)** — the same retrieval pipeline generalized to arbitrary, user-uploaded PDFs, for a more realistic, open-ended input setting.
- **LoRA fine-tuning** — adapted a small open-source LLM to follow a fixed answer format using LoRA, training roughly 0.4% of its parameters, with a genuine before/after evaluation including unseen questions.

Full write-ups, design decisions, and tradeoffs for each are in [GEN-AI-Projects](https://github.com/amruthathalla/GEN-AI-Projects), where more GenAI projects will continue to be added.

*I've also worked on a classical (non-LLM) ML text classifier as a point of comparison — included in the same repo, but the focus going forward is GenAI.*

## Find me

- LinkedIn:  www.linkedin.com/in/amrutha-varshini-thalla
- Email:  thallaamrutha5@gmail.com
- Portfolio:  https://amruthathallaportfolio.netlify.app/
