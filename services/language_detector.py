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

    def detect_language(self, text: str) -> str:
        """
        Detect the ISO language code for the given text.
        Returns 'en' as fallback.
        """
        if not text.strip():
            return "en"
            
        # 1. Whitelist for common English greetings and short phrases
        # FastText often misidentifies these short strings.
        english_greetings = {
            "hi", "hello", "hey", "greetings", "hi?", "hello?", "hey?",
            "thanks", "thank you", "thx", "cool", "ok", "okay", "got it",
            "bye", "goodbye", "see ya", "nice", "great", "awesome", "yep", "yes", "no",
            "how are you", "how are you doing", "how are you today", "how are you doing today",
            "what's up", "sup", "howdy", "k", "alright", "sure", "fine"
        }
        
        clean_text = text.lower().strip().rstrip("!?. ")
        if clean_text in english_greetings:
            logger.info(f"Language detection: Whitelisted English text: {text}")
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
            
            logger.info(f"Detected language: {lang_code} for text: {text[:50]}...")
            return lang_code
        except Exception as e:
            logger.error(f"Error during language detection: {e}")
            return "en"
