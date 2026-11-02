
import streamlit as st
from PIL import Image

from rag import (
    load_documents,
    split_text,
    search_documents,
    generate_answer
)


st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)


# LANGUAGE SETTINGS

LANGUAGES = {
    "English": "en",
    "Русский": "ru",
    "Türkmençe": "tk"
}

TEXTS = {
    "en": {
        "title": "Plant Disease Detection & AI Assistant",
        "language": "Select language",
        "vision": "1. Plant Disease Detection",
        "upload": "Upload a plant leaf image",
        "analyze": "Analyze Image",
        "prediction": "Prediction Results",
        "warning": "AI predictions are not confirmed diagnoses.",
        "rag": "2. Plant Disease AI Assistant",
        "description": "Ask a question about plant diseases. The assistant will search your documents and generate an answer.",
        "question": "Enter your question:",
        "ask": "Ask AI Assistant",
        "empty": "Please enter a question.",
        "searching": "Searching documents...",
        "generating": "Gemini is generating an answer...",
        "answer": "AI Generated Answer",
        "sources": "View source documents",
        "not_found": "No relevant information found in documents.",
        "image_error": "Image analysis error",
        "ai_error": "AI Assistant error",
        "caption": "Plant Disease AI | Computer Vision + RAG + Gemini"
    },

    "ru": {
        "title": "Определение болезней растений и ИИ-помощник",
        "language": "Выберите язык",
        "vision": "1. Определение болезней растений",
        "upload": "Загрузите фотографию листа растения",
        "analyze": "Анализировать изображение",
        "prediction": "Результаты определения",
        "warning": "Результаты ИИ не являются подтверждённым диагнозом.",
        "rag": "2. ИИ-помощник по болезням растений",
        "description": "Задайте вопрос о болезнях растений. Помощник выполнит поиск в документах и сформирует ответ.",
        "question": "Введите ваш вопрос:",
        "ask": "Спросить ИИ-помощника",
        "empty": "Пожалуйста, введите вопрос.",
        "searching": "Поиск в документах...",
        "generating": "Gemini формирует ответ...",
        "answer": "Ответ ИИ",
        "sources": "Показать исходные документы",
        "not_found": "В документах не найдена подходящая информация.",
        "image_error": "Ошибка анализа изображения",
        "ai_error": "Ошибка ИИ-помощника",
        "caption": "Plant Disease AI | Компьютерное зрение + RAG + Gemini"
    },

    "tk": {
        "title": "Ösümlik kesellerini anyklamak we emeli aň kömekçisi",
        "language": "Dili saýlaň",
        "vision": "1. Ösümlik kesellerini anyklamak",
        "upload": "Ösümlik ýapragynyň suratyny ýükläň",
        "analyze": "Suraty seljermek",
        "prediction": "Anyklama netijeleri",
        "warning": "Emeli aňyň netijesi gutarnykly diagnoz däldir.",
        "rag": "2. Ösümlik keselleri boýunça emeli aň kömekçisi",
        "description": "Ösümlik keselleri barada sorag beriň. Kömekçi resminamalardan maglumat gözläp, jogap taýýarlar.",
        "question": "Soragyňyzy giriziň:",
        "ask": "Emeli aň kömekçisinden soraň",
        "empty": "Haýyş, soragyňyzy giriziň.",
        "searching": "Resminamalardan gözlenýär...",
        "generating": "Gemini jogap taýýarlaýar...",
        "answer": "Emeli aňyň jogaby",
        "sources": "Çeşme resminamalary görkez",
        "not_found": "Resminamalarda degişli maglumat tapylmady.",
        "image_error": "Suraty seljermekde ýalňyşlyk",
        "ai_error": "Emeli aň kömekçisinde ýalňyşlyk",
        "caption": "Plant Disease AI | Kompýuter görüşi + RAG + Gemini"
    }
}


# DISEASE NAME TRANSLATIONS

DISEASE_TRANSLATIONS = {
    "en": {
        "Early blight": "Early blight",
        "Late blight": "Late blight",
        "healthy": "Healthy",
        "Bacterial spot": "Bacterial spot",
        "Leaf Mold": "Leaf Mold",
        "Septoria leaf spot": "Septoria leaf spot",
        "Spider mites": "Spider mites",
        "Target Spot": "Target Spot",
        "Yellow Leaf Curl Virus": "Yellow Leaf Curl Virus",
        "Mosaic virus": "Mosaic virus"
    },

    "ru": {
        "Early blight": "Ранняя пятнистость",
        "Late blight": "Фитофтороз",
        "healthy": "Здоровое растение",
        "Bacterial spot": "Бактериальная пятнистость",
        "Leaf Mold": "Листовая плесень",
        "Septoria leaf spot": "Септориоз",
        "Spider mites": "Паутинный клещ",
        "Target Spot": "Мишеневидная пятнистость",
        "Yellow Leaf Curl Virus": "Вирус жёлтой курчавости листьев",
        "Mosaic virus": "Вирус мозаики"
    },

    "tk": {
        "Early blight": "Irki tegmilli kesel",
        "Late blight": "Fitoftoroz",
        "healthy": "Sagdyn ösümlik",
        "Bacterial spot": "Bakterial tegmilli kesel",
        "Leaf Mold": "Ýaprak heň keseli",
        "Septoria leaf spot": "Septorioz",
        "Spider mites": "Öýjükli sakyrtga",
        "Target Spot": "Tegmilli kesel",
        "Yellow Leaf Curl Virus": "Ýapragyň sary towlanma wirusy",
        "Mosaic virus": "Mozaika wirusy"
    }
}


def translate_disease(label, language):
    translations = DISEASE_TRANSLATIONS.get(language, {})

    for english_name, translated_name in translations.items():
        if english_name.lower() in label.lower():
            return label.replace(
                english_name,
                translated_name
            )

    return label


# SELECT LANGUAGE

selected_language = st.selectbox(
    "🌐 Language / Язык / Dil",
    list(LANGUAGES.keys())
)

lang = LANGUAGES[selected_language]
t = TEXTS[lang]


st.title("🌿 " + t["title"])


# LOAD IMAGE CLASSIFIER

@st.cache_resource
def load_classifier():
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model="neghani/plantvillage-mobilenetv2",
        device=-1
    )


# LOAD KNOWLEDGE BASE

@st.cache_resource
def load_knowledge_base():
    documents = load_documents()
    return split_text(documents)


# PART 1: COMPUTER VISION

st.header(t["vision"])

uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption=t["upload"],
            use_container_width=True
        )

        if st.button(t["analyze"]):
            with st.spinner(t["searching"]):
                classifier = load_classifier()
                results = classifier(image, top_k=3)

            st.subheader(t["prediction"])

            for result in results:
                original_label = result["label"]

                translated_label = translate_disease(
                    original_label,
                    lang
                )

                st.write(
                    f"**{translated_label}** — "
                    f"{result['score'] * 100:.2f}%"
                )

            st.warning(t["warning"])

    except Exception as error:
        st.error(f"{t['image_error']}: {error}")


# PART 2: RAG + GEMINI

st.divider()

st.header(t["rag"])
st.write(t["description"])

question = st.text_input(
    t["question"],
    placeholder="What are the symptoms of early blight in tomato?"
)

if st.button(t["ask"], type="primary"):

    if not question.strip():
        st.warning(t["empty"])

    else:
        try:
            with st.spinner(t["searching"]):
                chunks = load_knowledge_base()
                results = search_documents(question, chunks)

            if not results:
                st.info(t["not_found"])

            else:
                with st.spinner(t["generating"]):

                    api_key = st.secrets["GEMINI_API_KEY"]

                    answer = generate_answer(
                        question,
                        results,
                        api_key,
                        language=lang
                    )

                st.subheader(t["answer"])
                st.write(answer)

                with st.expander(t["sources"]):

                    for result in results:
                        st.write(result["text"])

                        st.caption(
                            f"Source: {result['source']} | "
                            f"Similarity: {result['score']:.3f}"
                        )

        except Exception as error:
            st.error(f"{t['ai_error']}: {error}")


st.divider()

st.caption(t["caption"])
