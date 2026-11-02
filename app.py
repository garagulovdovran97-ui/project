
import re

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

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
        "vision": "1. Plant Disease Detection",
        "upload": "Upload a plant leaf image",
        "crop": "Select the leaf area using the rectangle",
        "analyze": "Analyze Selected Area",
        "leaf_prompt": "For best results, select a leaf and avoid fruit or background.",
        "prediction": "Prediction Results",
        "confidence": "Model confidence",
        "low": "Low confidence. The result is unreliable and should not be treated as a diagnosis.",
        "medium": "Moderate confidence. Additional verification is recommended.",
        "high": "High model confidence. The result still requires verification.",
        "warning": "AI predictions are not confirmed diagnoses.",
        "top3": "Top 3 Predictions",
        "analyzing": "Analyzing the selected area for possible plant diseases...",
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
        "vision": "1. Определение болезней растений",
        "upload": "Загрузите фотографию листа растения",
        "crop": "Выделите область листа прямоугольной рамкой",
        "analyze": "Проанализировать выделенную область",
        "leaf_prompt": "Для лучшего результата выделите лист, не включая плоды и фон.",
        "prediction": "Результаты определения",
        "confidence": "Уверенность модели",
        "low": "Низкая уверенность. Результат ненадёжен и не должен считаться диагнозом.",
        "medium": "Умеренная уверенность. Рекомендуется дополнительная проверка.",
        "high": "Высокая уверенность модели. Результат всё равно требует проверки.",
        "warning": "Результаты ИИ не являются подтверждённым диагнозом.",
        "top3": "Три наиболее вероятных результата",
        "analyzing": "Анализируем выделенную область на возможные болезни растения...",
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
        "vision": "1. Ösümlik kesellerini anyklamak",
        "upload": "Ösümlik ýapragynyň suratyny ýükläň",
        "crop": "Ýapragyň töweregini gönüburçluk bilen saýlaň",
        "analyze": "Saýlanan meýdany seljermek",
        "leaf_prompt": "Has gowy netije üçin, miweleri we fonuny goşmazdan, ýapragy saýlaň.",
        "prediction": "Anyklama netijeleri",
        "confidence": "Modeliň ynam derejesi",
        "low": "Ynam derejesi pes. Netije ygtybarly däl we diagnoz hökmünde kabul edilmeli däldir.",
        "medium": "Ynam derejesi ortaça. Goşmaça barlag maslahat berilýär.",
        "high": "Modeliň ynam derejesi ýokary. Şeýle-de bolsa, netije barlanmalydyr.",
        "warning": "Emeli aňyň netijesi gutarnykly diagnoz däldir.",
        "top3": "Iň ähtimal üç netije",
        "analyzing": "Saýlanan meýdanda ösümlik keselleri seljerilýär...",
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
        "healthy": "Здоровое состояние",
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
        "healthy": "Sagdyn ýagdaý",
        "Bacterial spot": "Bakterial tegmilli kesel",
        "Leaf Mold": "Ýaprak heň keseli",
        "Septoria leaf spot": "Septorioz",
        "Spider mites": "Öýjükli sakyrtga",
        "Target Spot": "Tegmilli kesel",
        "Yellow Leaf Curl Virus": "Ýapragyň sary towlanma wirusy",
        "Mosaic virus": "Mozaika wirusy"
    }
}


PLANT_TRANSLATIONS = {
    "en": {
        "apple": "Apple",
        "blueberry": "Blueberry",
        "cherry including sour": "Cherry",
        "corn maize": "Corn",
        "grape": "Grape",
        "orange": "Orange",
        "peach": "Peach",
        "pepper bell": "Bell pepper",
        "potato": "Potato",
        "raspberry": "Raspberry",
        "soybean": "Soybean",
        "squash": "Squash",
        "strawberry": "Strawberry",
        "tomato": "Tomato"
    },
    "ru": {
        "apple": "Яблоня",
        "blueberry": "Голубика",
        "cherry including sour": "Вишня",
        "corn maize": "Кукуруза",
        "grape": "Виноград",
        "orange": "Апельсин",
        "peach": "Персик",
        "pepper bell": "Болгарский перец",
        "potato": "Картофель",
        "raspberry": "Малина",
        "soybean": "Соя",
        "squash": "Кабачок",
        "strawberry": "Клубника",
        "tomato": "Томат"
    },
    "tk": {
        "apple": "Alma",
        "blueberry": "Gök miwe",
        "cherry including sour": "Ülje",
        "corn maize": "Mekgejöwen",
        "grape": "Üzüm",
        "orange": "Apelsin",
        "peach": "Şetdaly",
        "pepper bell": "Bolgar burçy",
        "potato": "Kartoşka",
        "raspberry": "Malina",
        "soybean": "Soya",
        "squash": "Kädi",
        "strawberry": "Ýertudana",
        "tomato": "Pomidor"
    }
}


def translate_disease(label, language):
    translations = DISEASE_TRANSLATIONS.get(language, {})
    parts = label.split("___", maxsplit=1)

    if len(parts) == 2:
        plant_name, disease_name = parts
        plant_name = re.sub(r"[_(),]+", " ", plant_name).strip()
        disease_name = re.sub(r"[_]+", " ", disease_name).strip()
        disease_name = re.sub(
            rf"^{re.escape(plant_name)}\s+",
            "",
            disease_name,
            flags=re.IGNORECASE
        )
    else:
        plant_name = ""
        disease_name = re.sub(r"[_]+", " ", label).strip()

    normalized_plant_name = re.sub(
        r"\s+",
        " ",
        plant_name.lower()
    )
    plant_name = PLANT_TRANSLATIONS.get(language, {}).get(
        normalized_plant_name,
        plant_name
    )

    for english_name, translated_name in translations.items():
        disease_name = re.sub(
            re.escape(english_name),
            translated_name,
            disease_name,
            flags=re.IGNORECASE
        )

    if plant_name:
        return f"{plant_name}: {disease_name}"

    return disease_name


# SELECT LANGUAGE

selected_language = st.selectbox(
    "🌐 Language / Язык / Dil",
    list(LANGUAGES.keys())
)

lang = LANGUAGES[selected_language]
t = TEXTS[lang]

st.title("🌿 " + t["title"])


# LOAD DISEASE CLASSIFIER

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
        original_image = Image.open(uploaded_file).convert("RGB")

        st.write(t["crop"])
        st.caption(t["leaf_prompt"])

        cropped_image = st_cropper(
            original_image,
            realtime_update=True,
            box_color="#00AA00",
            aspect_ratio=None,
            return_type="image"
        )

        if cropped_image is not None:

            if st.button(t["analyze"]):

                with st.spinner(t["analyzing"]):

                    classifier = load_classifier()

                    results = classifier(
                        cropped_image,
                        top_k=3
                    )

                if results:
                    results = sorted(
                        results,
                        key=lambda item: item["score"],
                        reverse=True
                    )

                    best_result = results[0]
                    confidence = best_result["score"] * 100

                    st.subheader(t["prediction"])
                    st.markdown(
                        f"### {translate_disease(best_result['label'], lang)}"
                    )
                    st.metric(
                        label=t["confidence"],
                        value=f"{confidence:.2f}%"
                    )
                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    if confidence < 60:
                        st.warning(t["low"])
                    elif confidence < 80:
                        st.info(t["medium"])
                    else:
                        st.success(t["high"])

                    st.subheader(t["top3"])
                    for result in results:
                        translated_name = translate_disease(
                            result["label"],
                            lang
                        )
                        st.write(
                            f"**{translated_name}** — "
                            f"{result['score'] * 100:.2f}%"
                        )

                    st.caption(t["warning"])

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

                results = search_documents(
                    question,
                    chunks
                )

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
