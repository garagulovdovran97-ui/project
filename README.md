# Plant Disease Detection & AI Assistant

## Project Overview

Plant Disease Detection & AI Assistant is an educational multilingual Streamlit application that combines computer vision, document search, and Gemini-generated answers.

Users can upload a plant image, select a region, check whether it appears to contain a leaf, and view possible disease classifications. They can also ask questions about plant diseases in English, Russian, or Turkmen; the app searches its local knowledge base and asks Gemini to answer using the retrieved passages.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- MobileNetV2
- Sentence Transformers
- Streamlit
- streamlit-cropper
- PyPDF
- Google Gemini API

## Main Features

### 1. Plant Disease Detection
- Upload an image and select the leaf area to analyze.
- Classify the selected image with the pre-trained `neghani/plantvillage-mobilenetv2` model and display the top three predictions and confidence scores.
- Show the interface and translated disease names in English, Russian, or Turkmen.

### 2. Knowledge Base Assistant (RAG)
- Read TXT and PDF documents.
- Split documents into text chunks.
- Convert text into semantic embeddings.
- Retrieve relevant information based on user questions.
- Generate a grounded answer with Gemini and show the retrieved source passages.

## Project Structure

```text
project/
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml  # Create locally; do not commit API keys
└── documents/
    └── plant_guide.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini API key to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Keep this file private. It is ignored by Git.

## Run Application

```bash
streamlit run app.py
```

## Limitations

- AI predictions are not guaranteed to be correct.
- Select a leaf area without fruit or background for the most useful prediction. The classifier returns the closest class it knows; it does not independently verify that the image contains a leaf.
- The classifier recognizes 38 PlantVillage classes. It may be unreliable on other plant species, image conditions, or diseases outside those classes.
- The confidence score is a model score, not a measured probability that the diagnosis is correct.
- The knowledge base contains only information provided in the documents.
- Gemini answers are based on retrieved passages and may still be inaccurate.
- Internet access is required for initial model downloads and Gemini API requests.

## Purpose

This project demonstrates the practical application of computer vision, deep learning, semantic search, multilingual interfaces, and Retrieval-Augmented Generation.