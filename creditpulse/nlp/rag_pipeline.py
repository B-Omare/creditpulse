"""
CreditPulse - RAG Pipeline
Phase 4: Builds a retrieval-augmented generation chatbot over CBK reports.
Loan officers can ask questions and get answers grounded in CBK documents.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

REPORTS_DIR = Path("data/cbk_reports")
CHROMA_DIR  = Path("data/chroma_db")


def build_vectorstore():
    """Load CBK documents, split into chunks, embed and store in ChromaDB."""
    print("Building RAG vector store...")

    # 1. Load all text files in the cbk_reports folder
    docs = []
    for txt_file in REPORTS_DIR.glob("*.txt"):
        print(f"  Loading {txt_file.name}...")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    print(f"  Loaded {len(docs)} document(s)")

    # 2. Split into chunks — each chunk is what the LLM will read
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    # 3. Embed using a free local model — no API cost
    print("  Loading embedding model (first time may take 1-2 minutes)...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store in ChromaDB — runs entirely on your local machine
    print("  Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        chunks,
        embedder,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"  Vector store saved to {CHROMA_DIR}")
    return vectorstore


def query_vectorstore(question, k=3):
    """Search the vector store for chunks relevant to a question."""
    print(f"\nSearching for: '{question}'")

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder
    )

    results = vectorstore.similarity_search(question, k=k)

    print(f"  Found {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"  ── Chunk {i} ──────────────────────────────")
        print(f"  {doc.page_content[:300]}...")
        print()

    return results


def main():
    # Build the vector store
    build_vectorstore()

    # Test with sample loan officer questions
    questions = [
        "What are the IFRS 9 loan classification stages?",
        "How does M-Pesa data predict creditworthiness?",
        "What are the causal drivers of default in East Africa?",
    ]

    for question in questions:
        query_vectorstore(question)

    print("RAG pipeline complete!")


if __name__ == "__main__":
    main()