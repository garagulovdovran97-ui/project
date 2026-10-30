# Plant Disease Detection & AI Assistant

## Project Overview

Plant Disease Detection & AI Assistant is an educational AI application that combines Computer Vision and Retrieval-Augmented Generation (RAG).

The application allows users to upload plant leaf images, identify possible diseases using a pre-trained deep learning model, and search information about plant diseases in a knowledge base.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- MobileNetV2
- Sentence Transformers
- Streamlit
- PyPDF

## Main Features

### 1. Plant Disease Detection
- Upload plant leaf images.
- Analyze images using a pre-trained classification model.
- Display the top three predictions and their confidence scores.

### 2. Knowledge Base Search (RAG)
- Read TXT and PDF documents.
- Split documents into text chunks.
- Convert text into semantic embeddings.
- Retrieve relevant information based on user questions.

## Project Structure

```text
project/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
└── documents/
    └── plant_guide.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Limitations

- AI predictions are not guaranteed to be correct.
- The knowledge base contains only information provided in the documents.
- The current RAG component retrieves relevant text but does not generate answers using a language model.
- Internet access is required to download the pre-trained models initially.

## Purpose

This project demonstrates the practical application of Computer Vision, Deep Learning, Semantic Search, and RAG concepts.