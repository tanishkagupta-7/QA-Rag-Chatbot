import os

import streamlit as st

from langchain_community.vectorstores import FAISS

from src.document_loader import load_uploaded_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.rag_pipeline import answer_question


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="centered",
)


# -----------------------------------
# Header
# -----------------------------------

st.title("📚 Document RAG Chatbot")

st.write(
    "Upload one or more PDF documents and ask questions about them."
)


# -----------------------------------
# Session state
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_file_signature" not in st.session_state:
    st.session_state.uploaded_file_signature = None


# -----------------------------------
# Upload documents
# -----------------------------------

uploaded_files = st.file_uploader(
    "📎 Upload PDF document(s)",
    type=["pdf"],
    accept_multiple_files=True,
)


# -----------------------------------
# Process documents only when changed
# -----------------------------------

if uploaded_files:

    current_signature = tuple(
        (file.name, file.size)
        for file in uploaded_files
    )

    if current_signature != st.session_state.uploaded_file_signature:

        with st.spinner(
            "Processing your document(s)..."
        ):

            documents = load_uploaded_documents(
                uploaded_files
            )

            chunks = split_documents(documents)

            embeddings = get_embeddings()

            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 12,
                    "lambda_mult": 0.7,
                },
            )

            st.session_state.retriever = retriever

            st.session_state.uploaded_file_signature = (
                current_signature
            )

        st.success(
            f"Processed {len(uploaded_files)} "
            f"document(s) into {len(chunks)} chunks."
        )


# -----------------------------------
# Clear conversation
# -----------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# -----------------------------------
# Show previous conversation
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.markdown("**Sources:**")

                for source in sources:

                    st.write(
                        f"📄 {source['filename']} "
                        f"— Page {source['page']}"
                    )

                    with st.expander(
                        "View source preview"
                    ):
                        st.write(
                            source["preview"]
                        )


# -----------------------------------
# Question input
# -----------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Show user message
    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------
    # Generate answer
    # --------------------------------

    with st.chat_message("assistant"):

        if st.session_state.retriever is None:

            answer = (
                "Please upload a PDF document first."
            )

            documents = []

        else:

            with st.spinner(
                "Searching your documents..."
            ):

                answer, documents = answer_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.messages,
                )


        st.markdown(answer)


        # --------------------------------
        # Build source information
        # --------------------------------

        sources = []

        for document in documents:

            source_path = document.metadata.get(
                "source",
                "Unknown document",
            )

            filename = os.path.basename(
                source_path
            )

            page = document.metadata.get(
                "page_label",
                "Unknown",
            )

            preview = document.page_content[:700]

            source = {
                "filename": filename,
                "page": page,
                "preview": preview,
            }

            if source not in sources:

                sources.append(source)


        # --------------------------------
        # Display sources
        # --------------------------------

        if sources:

            st.markdown("**Sources:**")

            for source in sources:

                st.write(
                    f"📄 {source['filename']} "
                    f"— Page {source['page']}"
                )

                with st.expander(
                    "View source preview"
                ):

                    st.write(
                        source["preview"]
                    )


    # --------------------------------
    # Save assistant message
    # --------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )