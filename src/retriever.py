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
        semantic_docs = self.semantic_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)

        scores = {}
        documents_by_id = {}

        rrf_k = 60

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

        ranked_documents = sorted(
            documents_by_id.items(),
            key=lambda item: scores[item[0]],
            reverse=True,
        )

        return [
            document
            for _, document in ranked_documents[:6]
        ]


def get_retriever():
    """Create the hybrid FAISS + BM25 retriever."""

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return HybridRetriever(vector_store)