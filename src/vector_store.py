from langchain_community.vectorstores import FAISS

from src.text_splitter import split_documents
from src.document_loader import load_documents
from src.embeddings import get_embeddings


VECTORSTORE_DIR = "vectorstore"


def create_vector_store():
    """Create a FAISS vector store from the PDF documents."""

    # 1. Load the PDF
    documents = load_documents()

    # 2. Split the documents into chunks
    chunks = split_documents(documents)

    print(f"Total chunks to embed: {len(chunks)}")

    # 3. Load the embedding model
    embeddings = get_embeddings()

    # 4. Convert chunks into embeddings and create FAISS index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # 5. Save FAISS index locally
    vector_store.save_local(VECTORSTORE_DIR)

    print(f"FAISS vector store saved to: {VECTORSTORE_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()