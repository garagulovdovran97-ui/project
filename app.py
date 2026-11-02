
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

from classification import (
    LEAF_DISEASE_MODEL_NAME,
    MODEL_NAME,
    prepare_predictions,
    translate_leaf_label,
    translate_produce,
)

from rag import (
    load_documents,
    split_text,
    search_documents,
    generate_answer
)


st.set_page_config(
    page_title="Fruit & Vegetable Recognition",
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
        "title": "Fruit, Vegetable & Leaf Health Assistant",
        "mode": "Choose analysis",
        "produce_mode": "Recognize a fruit or vegetable",
        "leaf_mode": "Check a leaf for possible disease",
        "produce_vision": "1. Fruit and Vegetable Recognition",
        "leaf_vision": "1. Possible Leaf Disease",
        "upload": "Upload a fruit or vegetable photo",
        "image_hint": "Draw a rectangle around one whole fruit or vegetable, excluding as much background as possible. Use a clear, well-lit photo. The model recognizes 36 classes; it does not diagnose plant diseases.",
        "image_crop": "Select one fruit or vegetable",
        "analyze": "Recognize Image",
        "prediction": "Recognition Result",
        "confidence": "Model score",
        "low": "The model is uncertain. Try a closer, clearer photo with less background.",
        "warning": "Suggestions are not guaranteed identifications. The model can still be confidently wrong on unfamiliar photos.",
        "top3": "Top 3 Matches",
        "analyzing": "Identifying the fruit or vegetable...",
        "leaf_upload": "Upload a clear photo of a plant leaf",
        "leaf_hint": "This mode only analyzes close-up photos of leaves; it cannot recognize whole fruits or vegetables. For an apple or other product, switch to “Recognize a fruit or vegetable.” Select one leaf area and exclude fruit and as much background as possible. The model recognizes 38 PlantVillage leaf/disease classes and may be wrong on other plants or photo conditions.",
        "leaf_analyze": "Check Leaf",
        "leaf_analyzing": "Checking the selected leaf for possible disease...",
        "leaf_prediction": "Possible leaf disease",
        "leaf_confidence": "Model score",
        "leaf_top3": "Top 3 possible leaf matches",
        "leaf_low": "The model is uncertain. Treat these as suggestions only, not a diagnosis.",
        "leaf_warning": "This model identifies patterns among its known leaf classes. It does not confirm the plant species or provide a medical/agricultural diagnosis.",
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
        "caption": "Fruit & Vegetable Recognition | Computer Vision + RAG + Gemini"
    },

    "ru": {
        "title": "Распознавание продуктов и анализ листьев",
        "mode": "Выберите анализ",
        "produce_mode": "Распознать овощ или фрукт",
        "leaf_mode": "Проверить лист на возможную болезнь",
        "produce_vision": "1. Распознавание овощей и фруктов",
        "leaf_vision": "1. Возможная болезнь листа",
        "upload": "Загрузите фото овоща или фрукта",
        "image_hint": "Обведите рамкой один целый овощ или фрукт, захватив как можно меньше фона. Используйте чёткое фото при хорошем освещении. Модель распознаёт 36 классов, но не определяет болезни растений.",
        "image_crop": "Выделите один овощ или фрукт",
        "analyze": "Распознать фото",
        "prediction": "Результат распознавания",
        "confidence": "Оценка модели",
        "low": "Модель не уверена в результате. Попробуйте сделать фото ближе и чётче, с меньшим количеством фона.",
        "warning": "Это предположение модели, а не гарантия правильного определения. На незнакомых фото модель может уверенно ошибаться.",
        "top3": "Три наиболее подходящих варианта",
        "analyzing": "Определяем овощ или фрукт...",
        "leaf_upload": "Загрузите чёткое фото листа растения",
        "leaf_hint": "Этот режим анализирует только крупные фото листьев и не распознаёт целые фрукты или овощи. Для яблока и других продуктов выберите режим «Распознать овощ или фрукт». Выделите один участок листа, исключив плоды и фон. Модель знает 38 классов листьев и болезней PlantVillage и может ошибаться на других растениях и условиях съёмки.",
        "leaf_analyze": "Проверить лист",
        "leaf_analyzing": "Проверяем выделенный участок листа на возможную болезнь...",
        "leaf_prediction": "Возможная болезнь листа",
        "leaf_confidence": "Оценка модели",
        "leaf_top3": "Три возможных варианта",
        "leaf_low": "Модель не уверена. Считайте результат лишь предположением, а не диагнозом.",
        "leaf_warning": "Модель сравнивает фото с известными ей классами листьев. Она не подтверждает вид растения и не ставит профессиональный диагноз.",
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
        "caption": "Распознавание овощей и фруктов | Компьютерное зрение + RAG + Gemini"
    },

    "tk": {
        "title": "Önümleri tanamak we ýapraklary seljermek",
        "mode": "Seljeriş görnüşini saýlaň",
        "produce_mode": "Gök önümi ýa-da miwäni tanamak",
        "leaf_mode": "Ýapragy mümkin bolan kesele barlamak",
        "produce_vision": "1. Gök önümleri we miweleri tanamak",
        "leaf_vision": "1. Ýaprakdaky mümkin bolan kesel",
        "upload": "Gök önümiň ýa-da miwäniň suratyny ýükläň",
        "image_hint": "Bir miwäni ýa-da gök önümi gönüburçluk bilen saýlaň we fonuň az bölegini goşuň. Aýdyň hem ýagty surat ulanyň. Model 36 görnüşi tanaýar, ösümlik kesellerini anyklamaýar.",
        "image_crop": "Bir miwäni ýa-da gök önümi saýlaň",
        "analyze": "Suraty tanamak",
        "prediction": "Tanamak netijesi",
        "confidence": "Modeliň bahasy",
        "low": "Model netijä ynamly däl. Has ýakyndan, aýdyň we fon az görünýän surat alyň.",
        "warning": "Netijeler diňe modeliň çaklamasydyr. Nätanyş suratlarda model ýokary ynam bilen hem ýalňyşyp biler.",
        "top3": "Iň laýyk üç wariant",
        "analyzing": "Gök önüm ýa-da miwe tanalýar...",
        "leaf_upload": "Ösümlik ýapragynyň aýdyň suratyny ýükläň",
        "leaf_hint": "Bu görnüş diňe ýapragyň ýakyndan düşürilen suratlaryny seljerýär; ol tutuş miweleri ýa-da gök önümleri tanamaýar. Alma ýa-da başga önüm üçin «Gök önümi ýa-da miwäni tanamak» görnüşini saýlaň. Bir ýapragy saýlaň we miweleri hem-de fonuny goşmaň. Model PlantVillage ulgamynyň 38 ýaprak/kesel synpyny tanaýar; başga ösümliklerde ýa-da surata düşüriş şertlerinde ýalňyşyp biler.",
        "leaf_analyze": "Ýapragy barlamak",
        "leaf_analyzing": "Saýlanan ýaprakda mümkin bolan kesel barlanýar...",
        "leaf_prediction": "Ýaprakdaky mümkin bolan kesel",
        "leaf_confidence": "Modeliň bahasy",
        "leaf_top3": "Iň ähtimal üç wariant",
        "leaf_low": "Model netijä ynamly däl. Ony diňe çaklama hökmünde kabul ediň, diagnoz däldir.",
        "leaf_warning": "Model suraty özüniň tanaýan ýaprak synplary bilen deňeşdirýär. Ol ösümligiň görnüşini tassyklamaýar we hünärmen diagnozyny goýmaýar.",
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
        "caption": "Gök önümleri we miweleri tanamak | Kompýuter görüşi + RAG + Gemini"
    }
}


# SELECT LANGUAGE

selected_language = st.selectbox(
    "🌐 Language / Язык / Dil",
    list(LANGUAGES.keys())
)

lang = LANGUAGES[selected_language]
t = TEXTS[lang]

st.title("🌿 " + t["title"])

analysis_mode = st.radio(
    t["mode"],
    [t["produce_mode"], t["leaf_mode"]],
    horizontal=True,
    key="analysis_mode",
)

# LOAD PRODUCE CLASSIFIER

@st.cache_resource
def load_classifier():
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )


@st.cache_resource
def load_leaf_classifier():
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=LEAF_DISEASE_MODEL_NAME,
        device=-1
    )


# LOAD KNOWLEDGE BASE

@st.cache_resource
def load_knowledge_base():
    documents = load_documents()
    return split_text(documents)


# PART 1: COMPUTER VISION

if analysis_mode == t["produce_mode"]:
    st.header(t["produce_vision"])
    uploaded_file = st.file_uploader(
        t["upload"],
        type=["jpg", "jpeg", "png", "webp"],
        key="produce_image",
    )

    if uploaded_file:
        try:
            original_image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")
            st.write(t["image_crop"])
            selected_produce = st_cropper(
                original_image,
                realtime_update=True,
                box_color="#00AA00",
                aspect_ratio=None,
                return_type="image",
            )
            st.caption(t["image_hint"])

            if selected_produce is not None and st.button(
                t["analyze"],
                type="primary",
                key="recognize_produce",
            ):
                with st.spinner(t["analyzing"]):
                    classifier = load_classifier()
                    results = prepare_predictions(
                        classifier(selected_produce, top_k=3)
                    )

                if results:
                    best_result = results[0]
                    confidence = best_result["score"] * 100

                    st.subheader(t["prediction"])
                    st.markdown(
                        f"### {translate_produce(best_result['label'], lang)}"
                    )
                    st.metric(
                        label=t["confidence"],
                        value=f"{confidence:.2f}%"
                    )
                    st.progress(min(max(confidence / 100, 0.0), 1.0))

                    if confidence < 50:
                        st.warning(t["low"])

                    st.subheader(t["top3"])
                    for result in results:
                        st.write(
                            f"**{translate_produce(result['label'], lang)}** — "
                            f"{result['score'] * 100:.2f}%"
                        )

                    st.caption(t["warning"])
                else:
                    st.error(t["image_error"])

        except Exception as error:
            st.error(f"{t['image_error']}: {error}")
else:
    st.header(t["leaf_vision"])
    st.info(t["leaf_hint"])
    uploaded_leaf = st.file_uploader(
        t["leaf_upload"],
        type=["jpg", "jpeg", "png", "webp"],
        key="leaf_image",
    )

    if uploaded_leaf:
        try:
            leaf_image = ImageOps.exif_transpose(
                Image.open(uploaded_leaf)
            ).convert("RGB")
            cropped_leaf = st_cropper(
                leaf_image,
                realtime_update=True,
                box_color="#00AA00",
                aspect_ratio=None,
                return_type="image",
            )

            if cropped_leaf is not None and st.button(
                t["leaf_analyze"],
                type="primary",
                key="analyze_leaf",
            ):
                with st.spinner(t["leaf_analyzing"]):
                    classifier = load_leaf_classifier()
                    leaf_results = prepare_predictions(
                        classifier(cropped_leaf, top_k=3)
                    )

                if leaf_results:
                    best_leaf_result = leaf_results[0]
                    leaf_confidence = best_leaf_result["score"] * 100
                    st.subheader(t["leaf_prediction"])
                    if leaf_confidence >= 60:
                        st.markdown(
                            "### "
                            + translate_leaf_label(
                                best_leaf_result["label"],
                                lang
                            )
                        )
                    else:
                        st.warning(t["leaf_low"])
                    st.metric(
                        label=t["leaf_confidence"],
                        value=f"{leaf_confidence:.2f}%"
                    )
                    st.progress(
                        min(max(leaf_confidence / 100, 0.0), 1.0)
                    )

                    st.subheader(t["leaf_top3"])
                    for result in leaf_results:
                        st.write(
                            f"**{translate_leaf_label(result['label'], lang)}** — "
                            f"{result['score'] * 100:.2f}%"
                        )
                    st.caption(t["leaf_warning"])
                else:
                    st.error(t["image_error"])

        except Exception as error:
            st.error(f"{t['image_error']}: {error}")


# PART 2: RAG + GEMINI

st.divider()

st.header(t["rag"])
st.write(t["description"])

question = st.text_input(
    t["question"],
    placeholder="What are common signs of leaf disease?"
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