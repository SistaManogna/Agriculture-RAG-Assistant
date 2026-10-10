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

# Agriculture RAG Assistant

A Streamlit agriculture assistant with a personalised welcome page, source-backed answers, follow-up questions, multilingual responses, and session-only PDF indexing.

## Run

Install `requirements.txt` in a Python environment and configure `OPENAI_API_KEY` in `.env`. Optionally set `OPENAI_MODEL`; the existing default is `gpt-5.6-luna`. The embedding model may download on first use.

```bash
venv/bin/python -m streamlit run app.py
```

After editing backend functions in `rag.py`, stop the running server with Ctrl+C and start it again. A browser refresh can rerun the UI while retaining an older imported backend module in the server process.

The repository includes `agriculture.index` and `chunks.pkl`. Run `ingest.py` to rebuild them from the dataset PDFs.

## Using the assistant

1. Enter a display name on the homepage.
2. Choose an answer language in the sidebar: English, Telugu, Hindi, Tamil, Kannada, or Malayalam.
3. Ask a question in a supported language, including native script or transliteration. Follow-ups use the latest three conversation exchanges to resolve the topic into a standalone English retrieval query.
4. Paragraph references such as `[1]` map to the numbered passages under **Sources**, with the original document name and page. References are validated against the supplied passages; this does not independently verify the factual accuracy of the model's interpretation.
5. To add documents, expand **Add your PDFs**, choose files, and click **Index selected PDFs**. Choose whether to search all documents, built-in documents, or uploaded PDFs only.

Uploads support up to five English text PDFs, 10 MB each, 100 pages and 300,000 extracted characters total. Scanned PDFs need OCR; encrypted PDFs are rejected. Changing the selected files invalidates their previous index until you index the new selection. Uploaded indexes are stored in session memory, never added to the built-in index. Retrieved passages and recent conversation text are sent to OpenAI to answer questions. Display names are added locally and not sent in model prompts.

New conversation clears messages while keeping the name and uploaded index. Change name / Home clears messages, name, and uploaded index. Reloading or ending a browser session may discard session data. Download conversation exports answers, reference numbers, and passages as Markdown.

## Implementation checks

```bash
venv/bin/python -m unittest discover -s tests -v
```

Tests cover UI flow, context forwarding, language selection, citation validation, PDF errors, and isolated retrieval. They mock model responses and are not an answer-quality evaluation dataset.
