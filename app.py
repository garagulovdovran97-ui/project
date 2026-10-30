import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection & AI Assistant")

st.write(
    "Upload a plant leaf image to identify possible diseases."
)

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="neghani/plantvillage-mobilenetv2",
        revision="v1.0",
        device=-1
    )

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            try:
                classifier = load_model()
                predictions = classifier(image, top_k=3)

                st.subheader("Analysis Results")

                for i, result in enumerate(predictions, start=1):
                    label = result["label"].replace("___", " — ")
                    confidence = result["score"] * 100

                    st.write(f"{i}. **{label}**")
                    st.progress(float(result["score"]))
                    st.write(f"Model score: {confidence:.2f}%")

                st.warning(
                    "This is an AI prediction, not a confirmed diagnosis. "
                    "Results may be inaccurate, especially for field photos. "
                    "Consult an agricultural expert before treating plants."
                )

            except Exception as error:
                st.error(f"An error occurred: {error}")

st.divider()

st.caption(
    "Educational project using an open-source image classification model."
)
