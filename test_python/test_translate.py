import os
import unittest
from api.translation import translate

os.environ["TRANSFORMERS_CACHE"] = "transformers_data"

german_versions = [
    {'provider': 'mbart_large_50', 'translation': 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt.'},
    {'provider': 'mbart_large_50', 'translation': 'Der Generalsekretär der Vereinten Nationen sagt, es gibt keine militärische Lösung in Syrien.'}
]

class TestTranslateMbartLarge50(unittest.TestCase):

    def test_de_to_en_GB(self):
        self.assertEqual(translate({
            "sentence": 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt',
            "language_from": "de_DE",
            "language_to": "en_GB",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), {'provider': 'mbart_large_50', 'translation': 'UN Secretary-General says there is no military solution in Syria'})

    def test_de_to_en_XX(self):
        self.assertEqual(translate({
            "sentence": 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt',
            "language_from": "de_DE",
            "language_to": "en_XX",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), {'provider': 'mbart_large_50', 'translation': 'UN Secretary-General says there is no military solution in Syria'})

    def test_en_GB_to_de(self):
        self.assertIn(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no military solution in Syria.',
            "language_from": "en_GB",
            "language_to": "de_DE",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), german_versions)

    def test_en_XX_to_de(self):
        self.assertIn(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no  military solution in Syria.',
            "language_from": "en_XX",
            "language_to": "de_DE",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), german_versions)

    def test_en_GB_to_en_GB_supported_but_strange(self):
        self.assertEqual(translate({
            "sentence": 'To meet',
            "language_from": "en_XX",
            "language_to": "en_XX",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }),{'provider': 'mbart_large_50', 'translation': 'ملاقات کرنے کے لئے'})

    def test_hu_HU_from_not_supported(self):
        self.assertEqual(translate({
            "sentence": 'Does not matter',
            "language_from": "hu_HU",
            "language_to": "en_GB",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), ['Not supported language_from hu_HU', 400])

    def test_hu_HU_to_not_supported(self):
        self.assertEqual(translate({
            "sentence": 'Does not matter',
            "language_from": "en_GB",
            "language_to": "hu_HU",
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }), ['Not supported language_to hu_HU', 400])

class TestTranslateGuessModel(unittest.TestCase):

    def test_en_to_de(self):
        self.assertEqual(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no  military solution in Syria.',
            "language_from": "en_GB",
            "language_to": "de_DE"
        }), {"translation": "Der Generalsekretär der Vereinten Nationen sagt, dass es in Syrien keine militärische Lösung gibt.", "provider": "t5"})

    def test_en_to_de_again_speedtest(self):
        self.assertEqual(translate({
            "sentence": 'Within this framework, individualism versus collectivism has turned out to be the most robust and persistent contrast between different cultures.',
            "language_from": "en_GB",
            "language_to": "de_DE"
        }), {"translation": 'Innerhalb dieses Rahmens erwies sich Individualismus gegen '
                'Kollektivismus als der robusteste und anhaltendste Kontrast '
                'zwischen verschiedenen Kulturen.', "provider": "t5"})

    def test_unknown_to_de(self):
        self.assertEqual(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no  military solution in Syria.',
            "language_from": "unknown",
            "language_to": "de_DE"
        }), ('No module found supporting language_from "unknown", language_to "de_DE"!',
 400))

    def test_en_to_unknown(self):
        self.assertEqual(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no  military solution in Syria.',
            "language_from": "en_GB",
            "language_to": "unknown"
        }), ('No module found supporting language_from "en_GB", language_to "unknown"!',
 400))

class TestTranslateOther(unittest.TestCase):

    def test_invalid_model(self):
        self.assertEqual(translate({
            "sentence": 'Does not matter',
            "language_from": "en_GB",
            "language_to": "hu_HU",
            "model": 'invalid model name'
        }), ('Module "invalid model name" not found!', 400))

if __name__ == '__main__':
    unittest.main()