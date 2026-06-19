from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="rag_papers")

def retrieve_chunks(question, n_results=4):
    question_embedding = model.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    
    return chunks, sources

if __name__ == "__main__":
    test_question = "What is the difference between naive RAG and modular RAG?"
    chunks, sources = retrieve_chunks(test_question)
    
    print(f"Question: {test_question}\n")
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        print(f"--- Result {i+1} (from {source}) ---")
        print(chunk[:300] + "...\n")