# 📚 Document RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions from user-uploaded PDF documents using semantic search, keyword retrieval, and an LLM.

## 🚀 Features

- Upload one or more PDF documents
- Extract and chunk document text
- Generate semantic embeddings using Hugging Face
- Store document embeddings in FAISS
- Hybrid retrieval using:
  - FAISS semantic search
  - BM25 keyword search
  - Reciprocal Rank Fusion (RRF)
- Conversational query rewriting for follow-up questions
- Generate grounded answers using Groq LLM
- Display retrieved document sources and page numbers
- Evaluation pipeline for retrieval and answer quality
- Streamlit-based interface

## 🏗️ Architecture

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Cleaning & Chunking
      ↓
Hugging Face Embeddings
      ↓
FAISS + BM25 Retrieval
      ↓
Reciprocal Rank Fusion
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Grounded Answer + Sources
      ↓
Streamlit App
