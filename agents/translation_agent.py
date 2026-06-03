"""Translation agent with skip/translate decision and findings-aware prompts."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from config import settings
from services.agent_findings import compose_translation_context, publish
from services.language_detector import LanguageDetector
from services.image_modality_policy import append_interactive_image_recommendation
from services.translation_policy import evaluate_translation_policy
from services.translator import Translator, _lang_name

logger = logging.getLogger(__name__)


class TranslationAgent:
    """Decide if translation is needed, then localize using session findings."""

    def __init__(
        self,
        llm: ChatOpenAI,
        llm_fast: ChatOpenAI | None = None,
        detector: LanguageDetector | None = None,
    ) -> None:
        self._llm = llm
        self._llm_fast = llm_fast or llm
        self._translator = Translator(llm)
        self._detector = detector or LanguageDetector()

    async def _nano_judge(
        self, response: str, target_lang: str
    ) -> str:
        label = _lang_name(target_lang)
        prompt = (
            f"Is this educational reply already fully acceptable in {label}?\n"
            f"Reply JSON only: {{\"action\": \"skip\"|\"translate\", \"reason\": \"...\"}}\n\n"
            f"Text:\n{response[:1500]}"
        )
        try:
            resp = await self._llm_fast.ainvoke(prompt)
            raw = (resp.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "", 1)
            parsed = json.loads(raw.strip())
            action = str(parsed.get("action", "translate")).lower()
            if action in ("skip", "translate"):
                return action
        except Exception as exc:
            logger.debug("Translation nano judge failed: %s", exc)
        return "translate"

    async def _translate_with_context(
        self, text: str, ctx: Any, subjects: str | None
    ) -> tuple[str, int, int]:
        label = _lang_name(ctx.target_lang)
        grade = ctx.student_grade or "B"
        glossary_note = (
            "Use simpler vocabulary and short sentences."
            if grade in ("C", "D")
            else "Use clear academic vocabulary."
        )
        visual_note = ""
        if ctx.visual_summary:
            visual_note = f"\nVisual context from session (for terminology consistency): {ctx.visual_summary}"

        prompt = (
            f"You are a professional educational translator for Indian school students.\n"
            f"Target language: **{label}**.\n"
            f"Subject context: {subjects or 'General'}.\n"
            f"{glossary_note}\n"
            "Preserve MathJax delimiters \\( \\) and \\[ \\] exactly.\n"
            "Preserve numbers and formulas.\n"
            "STEM terms may stay in English with a brief local gloss in parentheses when helpful.\n"
            f"{visual_note}\n"
            "Respond ONLY with the translated text.\n\n"
            f"Text:\n{text}"
        )
        resp = await self._llm.ainvoke(prompt)
        usage = getattr(resp, "usage_metadata", None) or getattr(
            resp, "response_metadata", {}
        ).get("token_usage", {})
        i_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        o_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return (resp.content or "").strip() or text, i_tokens, o_tokens

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = append_interactive_image_recommendation(
            state.get("response", "") or "", state
        )
        target_lang = state.get("language", "en")
        if state.get("language_preference_locked") and state.get(
            "preferred_response_language"
        ):
            target_lang = state.get("preferred_response_language")
        if not target_lang or str(target_lang).lower() == "null":
            target_lang = "en"

        min_lang_conf = float(settings.language_response_min_confidence or 0.80)
        query_conf = float(state.get("detected_language_confidence") or 0.0)
        if (
            not state.get("language_preference_locked")
            and str(target_lang).lower() != "en"
            and query_conf < min_lang_conf
        ):
            logger.info(
                "Translation target forced to English (query lang confidence %.3f < %.2f)",
                query_conf,
                min_lang_conf,
            )
            target_lang = "en"

        detected, confidence = self._detector.detect_language_with_confidence(response)

        decision = evaluate_translation_policy(
            response=response,
            target_lang=target_lang,
            detected_lang=detected,
            confidence=confidence,
            language_preference_locked=bool(state.get("language_preference_locked")),
            final_language=state.get("final_language"),
        )

        action = decision.action
        if action == "ambiguous" and settings.translation_use_nano_judge:
            action = await self._nano_judge(response, target_lang)
            decision.reason = f"Nano judge → {action}"

        findings = publish(
            state.get("agent_findings"),
            "translation_agent",
            "translation_policy",
            decision.reason,
            {
                "action": action,
                "reason": decision.reason,
                "target_lang": target_lang,
                "detected_lang": detected,
                "confidence": confidence,
            },
            priority=5,
        )

        updates: Dict[str, Any] = {
            "translation_action": action,
            "translation_skip_reason": decision.reason,
            "detected_response_language": detected,
            "detected_response_language_confidence": confidence,
            "agent_findings": findings,
        }

        if action == "skip":
            updates["response"] = response
            updates["final_language"] = detected or target_lang
            updates["is_translated"] = False
            return updates

        ctx = compose_translation_context(
            state.get("agent_findings"),
            target_lang=target_lang,
            detected_lang=detected,
            preference_locked=bool(state.get("language_preference_locked")),
        )
        subjects = ", ".join(ctx.subjects) if ctx.subjects else None

        if settings.translation_use_primary_model:
            translated, i_tokens, o_tokens = await self._translate_with_context(
                response, ctx, subjects
            )
            model_name = getattr(self._llm, "model_name", settings.model_name)
        else:
            translated, i_tokens, o_tokens = await self._translator.from_english(
                response, target_lang, context=subjects
            )
            model_name = getattr(self._translator._llm, "model_name", settings.translator_model_name)

        updates.update(
            {
                "response": translated,
                "final_language": target_lang,
                "is_translated": True,
                "llm_calls": 1,
                "input_tokens": i_tokens,
                "output_tokens": o_tokens,
                "aux_input_tokens": i_tokens,
                "aux_output_tokens": o_tokens,
                "token_usage_by_model": {
                    model_name: {"input": i_tokens, "output": o_tokens}
                },
            }
        )
        from services.costing import patch_component_usage

        updates.update(patch_component_usage("chat", model_name, i_tokens, o_tokens))
        return updates
