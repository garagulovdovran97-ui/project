# Fruit, Vegetable & Leaf Health Assistant

## Project Overview

This multilingual Streamlit application recognizes common fruits and vegetables, suggests possible diseases from plant-leaf photos, and answers plant-health questions using a local knowledge base and Gemini.

Choose between two separate image-analysis modes: whole fruit/vegetable recognition or possible leaf-disease classification. Users can also ask plant-health questions in English, Russian, or Turkmen; the app searches its local documents and asks Gemini to answer using the retrieved passages.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- ResNet-50
- MobileNetV2
- Sentence Transformers
- Streamlit
- streamlit-cropper
- PyPDF
- Google Gemini API

## Main Features

### 1. Fruit and Vegetable Recognition
- Upload a fruit or vegetable photo and select the item to reduce background interference.
- Classify it with the pre-trained [`dima806/fruit_vegetable_image_detection`](https://huggingface.co/dima806/fruit_vegetable_image_detection) ViT model and show up to three matches.
- Recognize 36 classes, including potato, tomato, apple, banana, carrot, cucumber, grapes, and watermelon.
- Show translated produce names in English, Russian, or Turkmen.
- In a separate mode, select a leaf area and get possible disease matches using the 38-class `neghani/plantvillage-mobilenetv2` model.

### 2. Plant Health Assistant (RAG)
- Read TXT and PDF documents.
- Split documents into text chunks.
- Convert text into semantic embeddings.
- Retrieve relevant information based on user questions.
- Generate a grounded answer with Gemini and show the retrieved source passages.

## Project Structure

```text
project/
├── app.py
├── classification.py
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

- The fruit-and-vegetable model recognizes only its 36 trained classes. It is not a general object detector and does not diagnose plant diseases.
- The leaf model recognizes only 38 PlantVillage classes and works on close-up leaf photos, not whole products. A fruit photo submitted in leaf mode can receive a plausible-looking but meaningless leaf label, so use fruit/vegetable recognition for whole products. It can also be inaccurate on unfamiliar plants, photo conditions, or diseases outside its training classes.
- Leaf predictions are suggestions, not confirmed diagnoses; confidence scores are not calibrated probabilities.
- In a quick check on 17 labeled potato images from a public test split, this model ranked potato first on 15; this small check is not a guarantee for independent phone photos.
- Model scores are not calibrated probabilities. Even a high score can be wrong, especially for blurry photos, unusual lighting, cluttered backgrounds, or items outside the supported classes.
- A clear, close photo of one whole item generally matches the model's training task better than a distant or cluttered scene.
- If an item is misclassified, select just that item with the crop rectangle and analyze again; the actual photo may differ from the model's training images.
- Gemini answers are based on retrieved passages and may still be inaccurate.
- Internet access is required for initial model downloads and Gemini API requests.

## Purpose

This project demonstrates practical computer vision, multilingual interfaces, semantic search, and Retrieval-Augmented Generation.
