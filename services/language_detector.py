"""Service for language detection using FastText."""

from __future__ import annotations

import logging
import os
from typing import Optional

import fasttext

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects the language of a given text using FastText."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "lid.176.bin")
        
        self.model_path = model_path
        self._model = None
        
        if not os.path.exists(self.model_path):
            logger.error(f"FastText model not found at {self.model_path}")
        else:
            try:
                self._model = fasttext.load_model(self.model_path)
                logger.info(f"FastText model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load FastText model: {e}")

    # Japanese greetings / common phrases to whitelist correctly as 'ja'
    _JAPANESE_GREETINGS = {
        "こんにちは", "こんばんは", "おはよう", "おはようございます",
        "ありがとう", "ありがとうございます", "すみません", "はい", "いいえ",
        "わかりました", "よろしく", "よろしくお願いします", "さようなら",
        "どうぞ", "お願いします",
    }

    @staticmethod
    def _contains_japanese_script(text: str) -> bool:
        """Return True if text contains Hiragana, Katakana, or CJK Kanji characters."""
        for ch in text:
            cp = ord(ch)
            if (
                0x3040 <= cp <= 0x309F   # Hiragana
                or 0x30A0 <= cp <= 0x30FF  # Katakana
                or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs (Kanji)
                or 0xFF65 <= cp <= 0xFF9F  # Halfwidth Katakana
            ):
                return True
        return False

    def detect_language(self, text: str) -> str:
        """
        Detect the ISO language code for the given text.
        Returns 'en' as fallback.
        """
        if not text.strip():
            return "en"

        # 1. Japanese script hard-detection — catches Hiragana/Katakana/Kanji
        #    before FastText which can mis-label short Japanese as 'zh' or 'ko'.
        if self._contains_japanese_script(text):
            logger.info("Language detection: Japanese script detected for: %s", text[:30])
            return "ja"

        # 2. Whitelist English greetings — FastText often misidentifies 'hii' etc.
        from services.utils import is_greeting
        if is_greeting(text):
            logger.info("Language detection: Whitelisted greeting text: %s", text)
            return "en"

        if not self._model:
            return "en"

        try:
            # Clean text (FastText works better with single lines)
            clean_text = text.replace("\n", " ").strip()
            predictions = self._model.predict(clean_text, k=1)
            # predictions is ([label], [prob])
            # labels are like '__label__en'
            label = predictions[0][0]
            lang_code = label.replace("__label__", "")

            logger.info("Detected language: %s for text: %s", lang_code, text[:50])
            return lang_code
        except Exception as e:
            logger.error("Error during language detection: %s", e)
            return "en"
