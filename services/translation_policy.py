"""Rules-first policy for skip vs translate localization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from config import settings

TranslationAction = Literal["skip", "translate", "ambiguous"]


@dataclass
class TranslationDecision:
    action: TranslationAction
    reason: str
    detected_lang: str
    confidence: float


_INDIC_LANGS = frozenset({"hi", "mr", "bn", "ta", "te", "kn", "gu", "pa", "ur"})


def _mixed_script_heuristic(text: str, target_lang: str) -> bool:
    """True if response likely mixes Latin with Indic script when target is Indic."""
    if target_lang not in _INDIC_LANGS:
        return False
    has_latin = bool(re.search(r"[A-Za-z]{4,}", text))
    has_indic = bool(re.search(r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF]", text))
    return has_latin and has_indic


def evaluate_translation_policy(
    *,
    response: str,
    target_lang: str,
    detected_lang: str,
    confidence: float,
    language_preference_locked: bool = False,
    final_language: Optional[str] = None,
) -> TranslationDecision:
    """Decide whether to skip translation, translate, or defer to nano judge."""
    text = (response or "").strip()
    target = (target_lang or "en").lower()
    detected = (detected_lang or "").lower()
    conf = float(confidence or 0.0)

    if not text:
        return TranslationDecision("skip", "Empty response", detected or target, conf)

    if final_language and final_language.lower() == target:
        return TranslationDecision(
            "skip",
            "Upstream already marked final_language",
            detected or target,
            conf,
        )

    skip_conf = settings.translation_match_skip_confidence
    translate_conf = settings.translation_mismatch_translate_confidence

    if target == "en":
        if detected == "en" and conf >= translate_conf:
            return TranslationDecision("skip", "Already English", "en", conf)
        if detected and detected != "en" and conf >= translate_conf:
            return TranslationDecision(
                "translate",
                f"Non-English response ({detected}) for English target",
                detected,
                conf,
            )
        if conf < translate_conf and len(text) < 40:
            return TranslationDecision("ambiguous", "Short English-target reply", detected, conf)
        return TranslationDecision("skip", "Assume English for en target", detected or "en", conf)

    if detected == target and conf >= skip_conf:
        return TranslationDecision(
            "skip",
            f"Already in target language ({target})",
            detected,
            conf,
        )

    if detected and detected != target and conf >= translate_conf:
        return TranslationDecision(
            "translate",
            f"Language mismatch: {detected} vs {target}",
            detected,
            conf,
        )

    if language_preference_locked and detected != target:
        return TranslationDecision(
            "translate",
            "Locked language preference mismatch",
            detected,
            conf,
        )

    if _mixed_script_heuristic(text, target):
        return TranslationDecision(
            "translate",
            "Mixed-script response for Indic target",
            detected,
            conf,
        )

    if 0.50 <= conf < skip_conf or len(text) < 20:
        return TranslationDecision("ambiguous", "Gray-zone detection", detected, conf)

    if target != "en":
        return TranslationDecision(
            "translate",
            f"Default translate to {target}",
            detected,
            conf,
        )

    return TranslationDecision("skip", "No translation needed", detected or target, conf)
