
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


LANGUAGE_INSTRUCTIONS = {
    "en": """
Write the entire answer in English.
Use natural, clear English.
Do not switch to Russian or Turkmen.
""",

    "ru": """
Отвечай исключительно на русском языке.
Весь ответ, включая заголовки, пояснения и выводы,
должен быть написан на русском языке.
Не используй английский или туркменский для объяснений.
Если исходные документы написаны на английском,
переводи содержащуюся в них информацию на русский язык.
Английские названия можно приводить в скобках,
если это необходимо для точности.
""",

    "tk": """
Jogabyň hemmesini diňe türkmen dilinde ýaz.
Düşündirişleri, sözbaşylary we netijeleri türkmen dilinde ber.
Rus ýa-da iňlis dillerinde düşündiriş ýazma.
Resminamalar başga dilde bolsa, ondaky maglumatlary
türkmen diline terjime et.
Zerur bolsa, ylmy atlary ýa-da tehniki adalgalary
ýaý içinde asyl görnüşinde görkez.
"""
}


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

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

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


def generate_answer(
    question,
    search_results,
    api_key,
    language="en"
):
    if not search_results:
        empty_messages = {
            "en": "No relevant information found in the documents.",
            "ru": "В документах не найдена подходящая информация.",
            "tk": "Resminamalarda degişli maglumat tapylmady."
        }

        return empty_messages.get(
            language,
            empty_messages["en"]
        )

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["en"]
    )

    client = genai.Client(
        api_key=api_key
    )

    context = "\n\n".join(
        [
            f"Source document: {item['source']}\n"
            f"Document content: {item['text']}"
            for item in search_results
        ]
    )

    prompt = f"""
You are a professional multilingual assistant.

STRICT RESPONSE LANGUAGE RULE:
{language_instruction}

The requested response language code is: {language}

You MUST follow the requested response language.
The language of the source documents does not determine
the language of your answer.

TASK:
Answer the user's question using the provided documents.

RULES:
1. Write the complete answer in the requested language.
2. Translate relevant information from the source documents
   into the requested language.
3. Do not copy long English passages into a Russian
   or Turkmen answer.
4. Use only information supported by the provided documents.
5. Never invent facts, symptoms, treatments, or recommendations.
6. If the documents do not contain enough information,
   clearly say so in the requested language.
7. Mention the source document in the requested language.
8. Keep the answer clear and easy to understand.
9. Do not change the requested response language
   even if the user asks in another language.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

FINAL INSTRUCTION:
Before responding, check that the entire answer is written
in the requested language. Correct any sentences that are
in the wrong language.

Now provide the answer.
"""

    errors = []

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.2
                }
            )

            answer = response.text

            if answer and answer.strip():
                return answer.strip()

            errors.append(
                f"{model_name}: empty response"
            )

        except Exception as error:
            errors.append(
                f"{model_name}: {str(error)}"
            )

    raise RuntimeError(
        "Не удалось получить ответ ни от одной модели Gemini.\n\n"
        + "\n\n".join(errors)
    )
