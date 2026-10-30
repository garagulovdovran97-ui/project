
import streamlit as st
from PIL import Image
from transformers import pipeline
from rag import load_documents, split_text, search_documents

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Detection & AI Assistant")

@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="neghani/plantvillage-mobilenetv2",
        device=-1
    )

@st.cache_resource
def load_knowledge_base():
    documents = load_documents()
    return split_text(documents)


# PART 1: IMAGE ANALYSIS

st.header("1. Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            try:
                classifier = load_classifier()
                results = classifier(image, top_k=3)

                st.subheader("Prediction Results")

                for result in results:
                    st.write(
                        f"**{result['label']}** — "
                        f"{result['score'] * 100:.2f}%"
                    )

                st.warning(
                    "AI predictions are not confirmed diagnoses."
                )

            except Exception as error:
                st.error(str(error))


# PART 2: TEXT SEARCH

st.divider()

st.header("2. Plant Disease AI Assistant")

st.write(
    "Ask a question about plant diseases. "
    "The system will search your TXT and PDF documents."
)

question = st.text_input(
    "Enter your question:",
    placeholder="What are the symptoms of early blight?"
)

if st.button("Search Knowledge Base"):
    if question.strip():
        try:
            with st.spinner("Searching documents..."):
                chunks = load_knowledge_base()
                results = search_documents(question, chunks)

            st.subheader("Information from Documents")

            if results:
                for result in results:
                    st.write(result["text"])

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Similarity: {result['score']:.3f}"
                    )
            else:
                st.info("No information found.")

        except Exception as error:
            st.error(str(error))
    else:
        st.warning("Please enter a question.")

st.divider()

st.caption(
    "Plant Disease AI | Computer Vision + Document Search"
)
