from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_loader import load_documents


def split_documents(documents):
    """Split documents into smaller chunks for retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print("-" * 60)
        print(chunks[0].page_content[:1000])
        print("-" * 60)

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)