
import os
import re

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from google import genai


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash"
]


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
        text = re.sub(
            r"\s+",
            " ",
            document["text"]
        ).strip()

        filename = document["filename"]

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


@st.cache_data
def encode_documents(texts):
    model = load_embedding_model()

    return model.encode(
        texts,
        convert_to_tensor=True
    )


def search_documents(question, chunks, top_k=3):
    if not chunks:
        return []

    model = load_embedding_model()

    question_vector = model.encode(
        question,
        convert_to_tensor=True
    )

    texts = [chunk["text"] for chunk in chunks]

    document_vectors = encode_documents(texts)

    scores = util.cos_sim(
        question_vector,
        document_vectors
    )[0]

    top_results = scores.topk(
        k=min(top_k, len(chunks))
    )

    results = []

    for index in top_results.indices:
        index = int(index)

        results.append({
            "text": chunks[index]["text"],
            "source": chunks[index]["source"],
            "score": float(scores[index])
        })

    return results


def generate_answer(question, search_results, api_key):
    if not search_results:
        return "В документах не найдена информация для ответа."

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        [
            f"Source: {item['source']}\n{item['text']}"
            for item in search_results
        ]
    )

    prompt = f"""
You are an AI assistant that answers questions using
the provided document context.

Rules:
1. Use only information from the provided context.
2. Do not invent facts.
3. If the documents do not contain enough information,
   clearly say so.
4. Answer in the same language as the user's question.
5. Mention the source document when possible.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    errors = []

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:
                return response.text

            errors.append(
                f"{model_name}: empty response"
            )

        except Exception as error:
            error_message = str(error)

            errors.append(
                f"{model_name}: {error_message}"
            )

            continue

    raise RuntimeError(
        "Не удалось получить ответ ни от одной модели Gemini.\n\n"
        + "\n\n".join(errors)
    )
