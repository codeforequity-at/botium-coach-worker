import os
import unittest
from api.translation import translate

os.environ["TRANSFORMERS_CACHE"] = "transformers_data"

german_versions = [
    {'translation': 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt.'},
    {'translation': 'Der Generalsekretär der Vereinten Nationen sagt, es gibt keine militärische Lösung in Syrien.'}
]

class TestTranslate(unittest.TestCase):

    def test_de_to_en_GB(self):
        self.assertEqual(translate({
            "sentence": 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt',
            "language_from": "de_DE",
            "language_to": "en_GB"
        }), {'translation': 'UN Secretary-General says there is no military solution in Syria'})

    def test_de_to_en_XX(self):
        self.assertEqual(translate({
            "sentence": 'Der Generalsekretär der Vereinten Nationen sagt, dass es keine militärische Lösung in Syrien gibt',
            "language_from": "de_DE",
            "language_to": "en_XX"
        }), {'translation': 'UN Secretary-General says there is no military solution in Syria'})

    def test_en_GB_to_de(self):
        self.assertIn(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no military solution in Syria.',
            "language_from": "en_GB",
            "language_to": "de_DE"
        }), german_versions)

    def test_en_XX_to_de(self):
        self.assertIn(translate({
            "sentence": 'The Secretary-General of the United Nations says there is no  military solution in Syria.',
            "language_from": "en_XX",
            "language_to": "de_DE"
        }), german_versions)

    def test_en_GB_to_en_GB_supported_but_strange(self):
        self.assertIn(translate({
            "sentence": 'To meet',
            "language_from": "en_XX",
            "language_to": "en_XX"
        }), 'ملاقات کرنے کے لئے')

    def test_hu_HU_from_not_supported(self):
        self.assertIn(translate({
            "sentence": 'Does not matter',
            "language_from": "hu_HU",
            "language_to": "en_GB"
        }), ['Not supported language_from hu_HU', 400])

    def test_hu_HU_to_not_supported(self):
        self.assertIn(translate({
            "sentence": 'Does not matter',
            "language_from": "en_GB",
            "language_to": "hu_HU"
        }), ['Not supported language_to hu_HU', 400])


if __name__ == '__main__':
    unittest.main()