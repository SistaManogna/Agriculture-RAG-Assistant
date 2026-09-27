import os
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DATASET_FOLDER = "datasets"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

model = SentenceTransformer("all-MiniLM-L6-v2")


# 1. LOAD PDF TEXT
documents = []

for filename in os.listdir(DATASET_FOLDER):

    if filename.endswith(".pdf"):

        filepath = os.path.join(DATASET_FOLDER, filename)

        print(f"Reading: {filename}")

        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:
                documents.append({
                    "text": text,
                    "source": filename,
                    "page": page_number + 1
                })


print(f"\nTotal pages loaded: {len(documents)}")


# 2. CREATE CHUNKS
chunks = []

for document in documents:

    text = document["text"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "source": document["source"],
            "page": document["page"]
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP


print(f"Total chunks created: {len(chunks)}")


# 3. CREATE EMBEDDINGS
texts = [chunk["text"] for chunk in chunks]

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# 4. CREATE FAISS DATABASE
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# 5. SAVE
faiss.write_index(index, "agriculture.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)


print("\n==============================")
print("RAG KNOWLEDGE BASE CREATED!")
print("==============================")
print(f"Chunks stored: {len(chunks)}")
print("Vector database: agriculture.index")
print("Metadata: chunks.pkl")