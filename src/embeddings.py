from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """Create and return the sentence-transformer embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    embeddings = get_embeddings()

    test_text = "What is Natural Language Processing?"

    vector = embeddings.embed_query(test_text)

    print("=" * 60)
    print("EMBEDDING MODEL READY")
    print("=" * 60)

    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")