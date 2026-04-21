import unittest

from services.language_detector import LanguageDetector


class LanguageDetectorResilienceTests(unittest.TestCase):
    def test_short_english_phrase_is_whitelisted(self):
        detector = LanguageDetector.__new__(LanguageDetector)
        detector._model = None
        detector.model_path = ""
        self.assertEqual(detector.detect_language("thats cool"), "en")

    def test_repeated_ack_variant_is_whitelisted(self):
        detector = LanguageDetector.__new__(LanguageDetector)
        detector._model = None
        detector.model_path = ""
        self.assertEqual(detector.detect_language("ohkkkk"), "en")


if __name__ == "__main__":
    unittest.main()
