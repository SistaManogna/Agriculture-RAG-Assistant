# 🌱 Agriculture RAG System

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project that uses a collection of agricultural documents to answer user questions using relevant information retrieved from the knowledge base.

The main goal of this project is to understand how **RAG works step by step** using agricultural information.

##  Objective

The project aims to build a simple AI system that can:

- Read agricultural documents
- Split the documents into smaller chunks
- Convert the chunks into embeddings
- Store the embeddings in a vector database
- Retrieve relevant information based on a user's question
- Use an LLM to generate an answer using the retrieved information

##  How RAG Works

The basic workflow is:

```text
Agricultural PDFs
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Embeddings
       ↓
Vector Database
       ↓
User Question
       ↓
Similarity Search
       ↓
Relevant Information
       ↓
LLM
       ↓
Final Answer
```


##  Knowledge Base

The project uses a small collection of agricultural documents related to topics such as:

- Soil
- Fertilizers
- Crops
- Nutrients
- Sustainable agriculture
- Farming practices

These documents act as the **knowledge base** for the RAG system.


##  Technologies Used

- **Python**
- **LangChain**
- **Sentence Transformers**
- **FAISS**
- **LLM**
- **PDF documents**


## Basic Project Structure

```text
agriculture-rag/
│
├── data/
│   ├── agriculture_documents/
│   │   ├── document1.pdf
│   │   ├── document2.pdf
│   │   └── ...
│
├── embeddings/
│
├── ingest.py
├── rag.py
├── requirements.txt
├── README.md
└── .gitignore
```

##  Example

### User Question

```text
What is the importance of nitrogen for crops?
```

### RAG System

The system searches the agricultural knowledge base and retrieves the most relevant document chunks.

### LLM Response

The retrieved information is given to the LLM, which generates an answer based on the available agricultural knowledge.


##  Project Goal

This project is primarily focused on **learning and implementing the basic RAG pipeline**.

Future versions can extend the system into a complete agricultural assistant with:

- Fertilizer recommendations
- Crop recommendations
- Soil analysis
- Weather information
- Crop disease detection
- Multilingual support


##  Author

**Manogna Sista**

B.Tech — CSE (AI & ML)

VNR Vignana Jyothi Institute of Engineering and Technology

---

## 📌 Note

This is an educational project created to understand the fundamentals of **Retrieval-Augmented Generation (RAG)** and its application in agriculture.
