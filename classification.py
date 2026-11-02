import re


MODEL_NAME = "dima806/fruit_vegetable_image_detection"
LEAF_DISEASE_MODEL_NAME = "neghani/plantvillage-mobilenetv2"

PRODUCE_TRANSLATIONS = {
    "ru": {
        "apple": "Яблоко",
        "banana": "Банан",
        "beetroot": "Свёкла",
        "bell pepper": "Болгарский перец",
        "cabbage": "Капуста",
        "capsicum": "Сладкий перец",
        "carrot": "Морковь",
        "cauliflower": "Цветная капуста",
        "chilli pepper": "Перец чили",
        "corn": "Кукуруза",
        "cucumber": "Огурец",
        "eggplant": "Баклажан",
        "garlic": "Чеснок",
        "ginger": "Имбирь",
        "grapes": "Виноград",
        "jalepeno": "Халапеньо",
        "kiwi": "Киви",
        "lemon": "Лимон",
        "lettuce": "Листовой салат",
        "mango": "Манго",
        "onion": "Лук",
        "orange": "Апельсин",
        "paprika": "Паприка",
        "pear": "Груша",
        "peas": "Горох",
        "pineapple": "Ананас",
        "pomegranate": "Гранат",
        "potato": "Картофель",
        "raddish": "Редис",
        "soy beans": "Соевые бобы",
        "spinach": "Шпинат",
        "sweetcorn": "Сладкая кукуруза",
        "sweetpotato": "Батат",
        "tomato": "Томат",
        "turnip": "Репа",
        "watermelon": "Арбуз",
    },
    "tk": {
        "apple": "Alma",
        "banana": "Banan",
        "beetroot": "Çugundyr",
        "bell pepper": "Bolgar burçy",
        "cabbage": "Kelem",
        "capsicum": "Süýji burç",
        "carrot": "Käşir",
        "cauliflower": "Reňkli kelem",
        "chilli pepper": "Çili burçy",
        "corn": "Mekgejöwen",
        "cucumber": "Hyýar",
        "eggplant": "Baklajan",
        "garlic": "Sarymsak",
        "ginger": "Zynjabil",
        "grapes": "Üzüm",
        "jalepeno": "Halapenyo",
        "kiwi": "Kiwi",
        "lemon": "Limon",
        "lettuce": "Salat ýapragy",
        "mango": "Mango",
        "onion": "Sogan",
        "orange": "Apelsin",
        "paprika": "Paprika",
        "pear": "Armut",
        "peas": "Nohut",
        "pineapple": "Ananas",
        "pomegranate": "Nar",
        "potato": "Kartoşka",
        "raddish": "Turp",
        "soy beans": "Soya noýbasy",
        "spinach": "Ysmanak",
        "sweetcorn": "Süýji mekgejöwen",
        "sweetpotato": "Batat",
        "tomato": "Pomidor",
        "turnip": "Şalgam",
        "watermelon": "Garpyz",
    },
}

ENGLISH_LABELS = {
    "jalepeno": "Jalapeño",
    "raddish": "Radish",
    "sweetcorn": "Sweet corn",
    "sweetpotato": "Sweet potato",
    "soy beans": "Soy beans",
}

LEAF_DISEASE_TRANSLATIONS = {
    "ru": {
        "early blight": "Ранняя пятнистость",
        "late blight": "Фитофтороз",
        "healthy": "Здоровое состояние",
        "bacterial spot": "Бактериальная пятнистость",
        "black rot": "Чёрная гниль",
        "cedar apple rust": "Ржавчина яблони",
        "powdery mildew": "Мучнистая роса",
        "common rust": "Обыкновенная ржавчина",
        "northern leaf blight": "Северный листовой ожог",
        "leaf mold": "Листовая плесень",
        "septoria leaf spot": "Септориоз",
        "spider mites two spotted spider mite": "Паутинный клещ",
        "target spot": "Мишеневидная пятнистость",
        "tomato yellow leaf curl virus": "Вирус жёлтой курчавости листьев",
        "tomato mosaic virus": "Вирус мозаики томата",
        "esca black measles": "Эска (чёрная корь)",
        "leaf blight isariopsis leaf spot": "Ожог листьев (пятнистость)",
        "cercospora leaf spot gray leaf spot": "Церкоспорозная пятнистость",
        "bacterial spot": "Бактериальная пятнистость",
        "leaf scorch": "Ожог листьев",
        "haunglongbing citrus greening": "Позеленение цитрусовых",
    },
    "tk": {
        "early blight": "Irki tegmilli kesel",
        "late blight": "Fitoftoroz",
        "healthy": "Sagdyn ýagdaý",
        "bacterial spot": "Bakterial tegmilli kesel",
        "black rot": "Gara çüýreme",
        "cedar apple rust": "Alma pos keseli",
        "powdery mildew": "Unjumak keseli",
        "common rust": "Adaty pos keseli",
        "northern leaf blight": "Ýapragyň demirgazyk ýanmasy",
        "leaf mold": "Ýaprak heň keseli",
        "septoria leaf spot": "Septorioz",
        "spider mites two spotted spider mite": "Öýjükli sakyrtga",
        "target spot": "Tegmilli kesel",
        "tomato yellow leaf curl virus": "Ýapragyň sary towlanma wirusy",
        "tomato mosaic virus": "Pomidor mozaika wirusy",
        "esca black measles": "Üzümiň esca keseli",
        "leaf blight isariopsis leaf spot": "Ýaprak tegmilli keseli",
        "cercospora leaf spot gray leaf spot": "Serkosporoz tegmilli kesel",
        "leaf scorch": "Ýapragyň ýanma keseli",
        "haunglongbing citrus greening": "Sitruslaryň sary keseli",
    },
}


def prepare_predictions(results, top_k=3):
    return sorted(
        results,
        key=lambda result: result["score"],
        reverse=True
    )[:top_k]


def translate_produce(label, language):
    normalized_label = re.sub(r"\s+", " ", label.strip().lower())

    if language == "en":
        return ENGLISH_LABELS.get(normalized_label, normalized_label.title())

    return PRODUCE_TRANSLATIONS.get(language, {}).get(
        normalized_label,
        ENGLISH_LABELS.get(normalized_label, normalized_label.title())
    )


def translate_leaf_label(label, language):
    parts = label.split("___", maxsplit=1)
    if len(parts) != 2:
        return label.replace("_", " ")

    raw_crop, raw_disease = parts
    crop = raw_crop.replace("_", " ").replace("(", "").replace(")", "")
    crop = re.sub(r"\s+", " ", crop).strip()
    crop_aliases = {
        "cherry including sour": "cherry including sour",
        "corn maize": "corn",
        "grape": "grapes",
        "pepper bell": "bell pepper",
    }
    crop = translate_produce(crop_aliases.get(crop.lower(), crop), language)

    normalized_disease = re.sub(
        r"[^a-z0-9]+",
        " ",
        raw_disease.lower()
    ).strip()
    disease = LEAF_DISEASE_TRANSLATIONS.get(language, {}).get(
        normalized_disease,
        re.sub(r"[_]+", " ", raw_disease).strip()
    )
    return f"{crop}: {disease}"
