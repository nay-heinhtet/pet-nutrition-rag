# Pet Nutrition RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions about pet nutrition using PDF documents as a knowledge base. Built with LangChain, ChromaDB, and Ollama.

## What It Does

This system takes PDF documents about pet nutrition, processes them into searchable chunks, and uses a local LLM to answer questions grounded in those documents. If the answer isn't in the documents, it says so instead of making something up.

## How It Works

1. **Load** — Reads PDF files from the `docs/` folder
2. **Chunk** — Splits documents into ~500 character pieces with overlap to preserve context
3. **Embed** — Converts each chunk into a numerical vector using `all-MiniLM-L6-v2` that captures its meaning
4. **Store** — Saves all vectors in a ChromaDB vector database for fast similarity search
5. **Retrieve** — When you ask a question, it finds the 3 most semantically similar chunks
6. **Generate** — Sends those chunks as context to Llama 3 (running locally via Ollama), which generates a grounded answer

## Example

```
Your question: How much water does a cat need daily?

Answer: According to the context, cats will drink approximately 2 milliliters
of water for every gram of dry food they eat.

Your question: What is the best smartphone to buy?

Answer: I don't have enough information to answer that.
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed

### Install Dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters chromadb sentence-transformers pypdf
```

### Download the LLM

```bash
ollama pull llama3:8b
```

### Add Documents

Place your PDF files about pet nutrition in a `docs/` folder in the project root.

### Run

```bash
python app.py
```

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | Framework for building the RAG pipeline |
| ChromaDB | Vector database for storing and searching document embeddings |
| Sentence Transformers | Embedding model (all-MiniLM-L6-v2) for converting text to vectors |
| Ollama + Llama 3 | Local LLM for generating answers |
| PyPDF | PDF text extraction |

## Limitations and Next Steps

- **PDF extraction quality**: Some PDFs produce text with extra spaces or broken words. A text cleaning step could improve retrieval accuracy.
- **Chunk size tuning**: The current 500-character chunk size works but could be optimized for different document types.
- **Better retrieval**: Could experiment with different embedding models or hybrid search (combining semantic search with keyword matching).
- **Web UI**: Adding a Streamlit or Gradio frontend would make it more user-friendly.
- **Evaluation**: Building a test set of questions with known answers to systematically measure accuracy.
