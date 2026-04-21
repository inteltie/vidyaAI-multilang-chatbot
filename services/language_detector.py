"""Service for language detection using FastText."""

from __future__ import annotations

import logging
import os
import re
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
    _SHORT_ENGLISH_CHAT_PATTERNS = (
        r"^(ok+|okay+|k+)$",
        r"^(thx+|thanks+|thank you)$",
        r"^(h+i+|hey+|hello+)$",
        r"^(y+e+s+|y+e+p+|n+o+)$",
        r"^(cool+|nice+|great+|awesome+)$",
        r"^that'?s cool$",
        r"^thats cool$",
        r"^(h+m+|h+mm+)$",
    )
    _AMBIGUOUS_SHORT_TEXT_MAX_WORDS = 3
    _AMBIGUOUS_SHORT_TEXT_MAX_CHARS = 24
    _LOW_CONFIDENCE_THRESHOLD = 0.70

    @staticmethod
    def _normalize_text(text: str) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip().lower())
        clean = re.sub(r"([a-z])\1{2,}", r"\1\1", clean)
        return clean

    def _is_short_english_chat(self, text: str) -> bool:
        norm = self._normalize_text(text)
        if not norm:
            return False
        if any(re.fullmatch(pat, norm) for pat in self._SHORT_ENGLISH_CHAT_PATTERNS):
            return True
        words = norm.split()
        # Bias tiny latin-script snippets to English to avoid false language flips.
        if (
            len(words) <= self._AMBIGUOUS_SHORT_TEXT_MAX_WORDS
            and len(norm) <= self._AMBIGUOUS_SHORT_TEXT_MAX_CHARS
            and re.fullmatch(r"[a-z\s'?!.,]+", norm)
        ):
            return True
        return False

    def detect_language_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Detect language code with confidence.
        Returns ('en', 0.0) on empty/error fallback.
        """
        if not text.strip():
            return "en", 0.0

        if self._contains_japanese_script(text):
            logger.info("Language detection: Japanese script detected for: %s", text[:30])
            return "ja", 1.0

        from services.utils import is_greeting
        if is_greeting(text):
            logger.info("Language detection: Greeting whitelist hit for: %s", text[:30])
            return "en", 0.95

        if not self._model:
            return "en", 0.0

        try:
            clean_text = text.replace("\n", " ").strip()
            labels, probs = self._model.predict(clean_text, k=1)
            label = labels[0]
            lang_code = label.replace("__label__", "")
            confidence = float(probs[0]) if probs else 0.0
            logger.info(
                "Detected language: %s (confidence=%.3f) for text: %s",
                lang_code,
                confidence,
                text[:50],
            )
            return lang_code, confidence
        except Exception as e:
            logger.error("Error during language detection: %s", e)
            return "en", 0.0

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
        detected, confidence = self.detect_language_with_confidence(text)
        # Keep deterministic fallback for short low-confidence snippets.
        if (
            detected != "en"
            and self._is_short_english_chat(text)
            and confidence < self._LOW_CONFIDENCE_THRESHOLD
        ):
            return "en"
        return detected
