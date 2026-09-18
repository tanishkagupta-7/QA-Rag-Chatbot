from src.retriever import get_retriever
from src.llm import generate_response, rewrite_query


def answer_question(question, retriever=None, chat_history=None):
    """Answer a question using conversational document retrieval."""

    # Use empty history when no history is provided
    if chat_history is None:
        chat_history = []

    # 1. Use the provided retriever if available.
    # Otherwise, load the existing saved retriever.
    if retriever is None:
        retriever = get_retriever()

    # 2. Rewrite the question using conversation history
    standalone_question = rewrite_query(
        question,
        chat_history,
    )

    print(f"\nOriginal question: {question}")
    print(f"Standalone question: {standalone_question}")

    # 3. Retrieve relevant chunks using the rewritten question
    documents = retriever.invoke(standalone_question)

    # 4. Combine retrieved chunks into context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # 5. Create the RAG prompt
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Follow these rules:
1. Give a direct answer to exactly what the user asked.
2. Prefer definitions and statements directly supported by the context.
3. Do not add related information unless it is necessary to answer the question.
4. Do not use outside knowledge.
5. If multiple pieces of context are relevant, combine them into one clear answer.
6. Keep the answer concise and focused.
7. If the context does not contain enough information to answer the question, say:
"I don't know based on the provided document."

Context:
----------------
{context}
----------------

Conversation history:
----------------
{chat_history}
----------------

User question:
{question}

Answer:
"""

    # 6. Generate the final answer
    answer = generate_response(prompt)

    return answer, documents


if __name__ == "__main__":

    question = "What is Natural Language Processing?"

    answer, documents = answer_question(question)

    print("=" * 60)
    print("RAG QA TEST")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for document in documents:
        print(
            f"- Page {document.metadata.get('page_label')}"
        )
