import os
from sentence_transformers import SentenceTransformer
import chromadb

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="rag_papers")

text_folder = "extracted_text"
chunk_id = 0

for filename in os.listdir(text_folder):
    if filename.endswith(".txt"):
        path = os.path.join(text_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"{filename}: split into {len(chunks)} chunks")

        embeddings = model.encode(chunks).tolist()

        ids = [f"chunk_{chunk_id + i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        chunk_id += len(chunks)

print(f"\nDone. Total chunks stored: {collection.count()}")