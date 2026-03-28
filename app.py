"""
Pet Nutrition RAG System
A Retrieval-Augmented Generation system that answers questions
about pet nutrition using PDF documents as a knowledge base.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

# ── Step 1: Load PDF documents ──────────────────────────────────────
print("Loading documents...")
loader = DirectoryLoader('docs', glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ── Step 2: Split documents into chunks ─────────────────────────────
print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ── Step 3: Create embeddings and store in vector database ──────────
print("Creating vector store (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./chroma_db')
print("Vector store created")

# ── Step 4: Set up retriever ────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ── Step 5: Set up LLM ─────────────────────────────────────────────
llm = ChatOllama(model='llama3:8b', temperature=0)

# ── RAG function ────────────────────────────────────────────────────
def ask_question(question):
    """
    Takes a question, retrieves relevant chunks from the vector store,
    builds a prompt with those chunks as context, and sends it to the
    LLM to generate a grounded answer.
    """
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Combine chunks into one context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build the prompt
    prompt = f"""Use the following context to answer the question. 
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # Send to LLM and return the response
    response = llm.invoke(prompt)
    return response.content

# ── Interactive loop ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Pet Nutrition Knowledge Base")
    print("Ask any question about pet nutrition.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        if not question.strip():
            continue
        print(f"\nAnswer: {ask_question(question)}")
