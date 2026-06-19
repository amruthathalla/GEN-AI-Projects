import os
from groq import Groq
from retrieve import retrieve_chunks

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(question, n_results=4):
    chunks, sources = retrieve_chunks(question, n_results)
    
    context = ""
    for chunk, source in zip(chunks, sources):
        context += f"[Source: {source}]\n{chunk}\n\n"
    
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    answer = response.choices[0].message.content
    return answer, sources

if __name__ == "__main__":
    test_question = "What is the difference between naive RAG and modular RAG?"
    answer, sources = generate_answer(test_question)
    
    print(f"Question: {test_question}\n")
    print(f"Answer:\n{answer}\n")
    print(f"Sources used: {set(sources)}")