import os
from dotenv import load_dotenv
from groq import Groq
from source_text import SOURCE_TEXT

load_dotenv(override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_generative(question: str) -> str:
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say so clearly instead of guessing.

Context:
{SOURCE_TEXT}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    test_question = "How much does the Professional plan cost per guard?"
    answer = answer_generative(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")