import unittest

from classification import (
    LEAF_DISEASE_MODEL_NAME,
    PRODUCE_TRANSLATIONS,
    prepare_predictions,
    translate_leaf_label,
    translate_produce,
)


class PreparePredictionsTests(unittest.TestCase):
    def test_sorts_and_limits_model_predictions(self):
        results = [
            {"label": "potato", "score": 0.1},
            {"label": "tomato", "score": 0.8},
            {"label": "apple", "score": 0.3},
            {"label": "banana", "score": 0.2},
        ]

        self.assertEqual(
            prepare_predictions(results, top_k=3),
            [
                {"label": "tomato", "score": 0.8},
                {"label": "apple", "score": 0.3},
                {"label": "banana", "score": 0.2},
            ]
        )

    def test_supports_all_36_model_labels_in_each_language(self):
        labels = {
            "apple", "banana", "beetroot", "bell pepper", "cabbage",
            "capsicum", "carrot", "cauliflower", "chilli pepper", "corn",
            "cucumber", "eggplant", "garlic", "ginger", "grapes",
            "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion",
            "orange", "paprika", "pear", "peas", "pineapple", "pomegranate",
            "potato", "raddish", "soy beans", "spinach", "sweetcorn",
            "sweetpotato", "tomato", "turnip", "watermelon",
        }

        self.assertEqual(set(PRODUCE_TRANSLATIONS["ru"]), labels)
        self.assertEqual(set(PRODUCE_TRANSLATIONS["tk"]), labels)
        self.assertEqual(len(labels), 36)

    def test_translates_potato_and_normalizes_model_typo(self):
        self.assertEqual(translate_produce("potato", "ru"), "Картофель")
        self.assertEqual(translate_produce("potato", "tk"), "Kartoşka")
        self.assertEqual(translate_produce("raddish", "en"), "Radish")
        self.assertEqual(translate_produce("sweetpotato", "ru"), "Батат")

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(translate_produce("tomato", "fr"), "Tomato")

    def test_leaf_disease_labels_include_crop_and_disease(self):
        self.assertEqual(
            translate_leaf_label("Potato___Late_blight", "ru"),
            "Картофель: Фитофтороз"
        )
        self.assertEqual(
            translate_leaf_label("Tomato___healthy", "tk"),
            "Pomidor: Sagdyn ýagdaý"
        )
        self.assertEqual(
            translate_leaf_label("Grape___Black_rot", "en"),
            "Grapes: Black rot"
        )

    def test_leaf_disease_classifier_is_separate_from_produce_classifier(self):
        self.assertNotEqual(LEAF_DISEASE_MODEL_NAME, "dima806/fruit_vegetable_image_detection")


if __name__ == "__main__":
    unittest.main()
