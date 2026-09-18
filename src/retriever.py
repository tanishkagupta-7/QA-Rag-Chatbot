from hashlib import md5

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from src.embeddings import get_embeddings


VECTORSTORE_DIR = "vectorstore"


class HybridRetriever:
    """Hybrid retriever combining FAISS semantic search and BM25 keyword search."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

        self.semantic_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

        documents = list(
            vector_store.docstore._dict.values()
        )

        self.keyword_retriever = BM25Retriever.from_documents(
            documents
        )

        self.keyword_retriever.k = 8

    def invoke(self, query):
        """Retrieve documents using FAISS + BM25 and combine with RRF."""

        semantic_docs = self.semantic_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)

        scores = {}
        documents_by_id = {}

        rrf_k = 60

        # Semantic ranking
        for rank, document in enumerate(
            semantic_docs,
            start=1,
        ):
            document_id = md5(
                document.page_content.encode("utf-8")
            ).hexdigest()

            documents_by_id[document_id] = document

            scores[document_id] = (
                scores.get(document_id, 0)
                + 1 / (rrf_k + rank)
            )

        # Keyword ranking
        for rank, document in enumerate(
            keyword_docs,
            start=1,
        ):
            document_id = md5(
                document.page_content.encode("utf-8")
            ).hexdigest()

            documents_by_id[document_id] = document

            scores[document_id] = (
                scores.get(document_id, 0)
                + 1 / (rrf_k + rank)
            )

        # Combined RRF ranking
        ranked_documents = sorted(
            documents_by_id.items(),
            key=lambda item: scores[item[0]],
            reverse=True,
        )

        # Return top 6 documents
        return [
            document
            for _, document in ranked_documents[:6]
        ]


def get_retriever():
    """Load the saved FAISS vector store and create a hybrid retriever."""

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return HybridRetriever(vector_store)


if __name__ == "__main__":

    retriever = get_retriever()

    query = "What is Natural Language Processing?"

    results = retriever.invoke(query)

    print("=" * 60)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 60)

    print(f"Query: {query}")
    print(f"Relevant chunks found: {len(results)}")

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content[:500])
        print(
            f"Page: {document.metadata.get('page_label')}"
        )