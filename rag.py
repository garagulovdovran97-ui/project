
import os
import re

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def load_documents():
    documents = []
    folder = "documents"

    if not os.path.exists(folder):
        return documents

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

        elif filename.lower().endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:
            continue

        if text.strip():
            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


def split_text(documents, chunk_size=500, overlap=100):
    chunks = []

    for document in documents:
        text = document["text"]
        filename = document["filename"]

        text = re.sub(r"\s+", " ", text).strip()

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": filename
                })

            start += chunk_size - overlap

    return chunks


def search_documents(question, chunks, top_k=3):
    if not chunks:
        return []

    model = load_embedding_model()

    question_vector = model.encode(
        question,
        convert_to_tensor=True
    )

    texts = [chunk["text"] for chunk in chunks]

    document_vectors = model.encode(
        texts,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_vector,
        document_vectors
    )[0]

    results = []

    top_results = scores.topk(
        k=min(top_k, len(chunks))
    )

    for index in top_results.indices:
        index = int(index)

        results.append({
            "text": chunks[index]["text"],
            "source": chunks[index]["source"],
            "score": float(scores[index])
        })

    return results
