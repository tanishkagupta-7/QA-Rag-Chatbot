from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    """Load all PDF documents from the documents directory."""

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in data/documents/"
        )

    documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()

        documents.extend(pdf_documents)

        print(f"Pages loaded: {len(pdf_documents)}")

    return documents


def load_uploaded_documents(uploaded_files):
    """Load PDF documents uploaded through Streamlit."""

    documents = []

    for uploaded_file in uploaded_files:

        temp_path = Path("data") / uploaded_file.name

        temp_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        print(f"Loading uploaded file: {uploaded_file.name}")

        loader = PyPDFLoader(str(temp_path))
        pdf_documents = loader.load()

        for document in pdf_documents:
            document.metadata["source"] = uploaded_file.name

        documents.extend(pdf_documents)

        temp_path.unlink()

        print(
            f"Pages loaded from {uploaded_file.name}: "
            f"{len(pdf_documents)}"
        )

    return documents