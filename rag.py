from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def load_documents():
    documents = []
    folder = "documents"

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as file:
                documents.append(file.read())

        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            documents.append(text)

    return documents


def split_text(documents):
    chunks = []

    for document in documents:
        paragraphs = document.split("\n")

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if len(paragraph) > 30:
                chunks.append(paragraph)

    return chunks


def search_documents(question, chunks):
    if not chunks:
        return []

    question_vector = model.encode(
        question,
        convert_to_tensor=True
    )

    document_vectors = model.encode(
        chunks,
        convert_to_tensor=True
    )

    from sentence_transformers import util

    scores = util.cos_sim(
        question_vector,
        document_vectors
    )[0]

    top_results = scores.topk(
        k=min(3, len(chunks))
    )

    return [
        {
            "text": chunks[index],
            "score": float(scores[index])
        }
        for index in top_results.indices
    ]
