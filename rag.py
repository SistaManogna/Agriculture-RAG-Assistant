import os
import pickle
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

index = faiss.read_index("agriculture.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(query, top_k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        result = chunks[index_position]

        results.append({
            "text": result["text"],
            "source": result["source"],
            "page": result["page"],
            "distance": float(distance)
        })

    return results


# ==========================================
# GENERATE ANSWER USING OPENAI
# ==========================================

def generate_answer(question, retrieved_chunks):

    context = ""

    for i, chunk in enumerate(retrieved_chunks):

        context += f"""
SOURCE {i + 1}
Document: {chunk['source']}
Page: {chunk['page']}

Content:
{chunk['text']}

-------------------------
"""


    prompt = f"""
You are an Agriculture Research Assistant.

Answer the user's question using ONLY the information
provided in the retrieved documents below.

If the retrieved documents do not contain enough information
to answer the question, say:

"I couldn't find enough information in the provided documents."

Do not invent facts.

Give a clear and student-friendly answer.

At the end, mention the source documents and page numbers
used for the answer.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}
"""


    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


# ==========================================
# TEST THE RAG SYSTEM
# ==========================================

if __name__ == "__main__":

    question = input("\nAsk an agriculture question: ")

    print("\nSearching knowledge base...")

    retrieved_chunks = retrieve(question)

    print("Relevant information retrieved.")

    print("\nGenerating answer...\n")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("================================")
    print("AGRICULTURE RAG ASSISTANT")
    print("================================")

    print("\nAnswer:\n")
    print(answer)

    print("\n\nRetrieved Sources:")

    for chunk in retrieved_chunks:

        print(
            f"- {chunk['source']} "
            f"(Page {chunk['page']})"
        )