import unittest
from unittest.mock import patch

from nodes.analyze_query import AnalyzeQueryNode
from nodes.retrieve_documents import RetrieveDocumentsNode
from services.query_classifier import QueryClassifier


class _FakeDetector:
    def detect_language(self, _query: str) -> str:
        return "en"

    def detect_language_with_confidence(self, _query: str):
        return "en", 0.99


class _FakeClassifier:
    def __init__(self, quality):
        self._quality = quality
        self._llm = None

    def assess_query_quality(self, _query: str):
        return self._quality

    def _check_heuristics(self, _query: str):
        return None

    async def analyze(self, query, _history):
        from services.query_classifier import QueryClassification

        return QueryClassification(
            query_type="curriculum_specific",
            translated_query=query,
            confidence=0.8,
            reasoning="test",
            subjects=["General"],
        )


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    async def retrieve(self, query_en, filters=None, intent=None):
        return self._docs


class _FakeWebSearchTool:
    async def execute(self, query):
        return "web result", 2, 3


class QueryGuardrailsTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_query_greeting_bypasses_unclear_clarification(self):
        from services.query_classifier import QueryQualityAssessment, QueryClassification

        class _GreetingClassifier(_FakeClassifier):
            def _check_heuristics(self, _query: str):
                return QueryClassification(
                    query_type="conversational",
                    translated_query="hi",
                    confidence=1.0,
                    reasoning="Matched social heuristic (greeting/ack).",
                    subjects=["General"],
                )

        quality = QueryQualityAssessment(
            is_unclear=True,
            reason="noisy_or_gibberish_query",
            normalized_query="hi",
            clarification_prompt="Please clarify",
        )
        node = AnalyzeQueryNode(_GreetingClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "hi",
                "conversation_history": [],
                "context_gathering_attempts": 0,
            }
        )

        self.assertEqual(updates["query_type"], "conversational")
        self.assertEqual(updates["query_en"], "hi")
        self.assertNotIn("response", updates)
        self.assertIsNone(updates["clarification_message"])

    async def test_analyze_query_asks_single_clarification_for_unclear_input(self):
        from services.query_classifier import QueryQualityAssessment

        quality = QueryQualityAssessment(
            is_unclear=True,
            reason="ambiguous_abbreviation_query",
            normalized_query="physics help",
            clarification_prompt="Can you clarify what you mean?",
        )
        node = AnalyzeQueryNode(_FakeClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "phy pls",
                "conversation_history": [],
                "context_gathering_attempts": 0,
            }
        )

        self.assertEqual(updates["query_type"], "conversational")
        self.assertEqual(updates["response"], "Can you clarify what you mean?")
        self.assertEqual(updates["context_gathering_attempts"], 1)
        self.assertTrue(updates["needs_context"])

    async def test_analyze_query_after_one_attempt_uses_best_effort(self):
        from services.query_classifier import QueryQualityAssessment

        quality = QueryQualityAssessment(
            is_unclear=True,
            reason="noisy_or_gibberish_query",
            normalized_query="asdf",
            clarification_prompt="Please clarify",
        )
        node = AnalyzeQueryNode(_FakeClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "asdf",
                "conversation_history": [],
                "context_gathering_attempts": 1,
            }
        )

        self.assertEqual(updates["query_type"], "curriculum_specific")
        self.assertEqual(updates["query_en"], "asdf")

    async def test_analyze_query_detects_memory_intent_name(self):
        from services.query_classifier import QueryQualityAssessment

        quality = QueryQualityAssessment(
            is_unclear=False,
            reason=None,
            normalized_query="what is my name",
            clarification_prompt=None,
        )
        node = AnalyzeQueryNode(_FakeClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "what is my name",
                "conversation_history": [],
                "context_gathering_attempts": 0,
                "user_session_id": "memory-001",
            }
        )

        self.assertEqual(updates["query_type"], "conversational")
        self.assertEqual(updates["memory_intent"], "name")
        self.assertEqual(updates["query_en"], "what is my name")

    async def test_analyze_query_detects_memory_intent_class_level(self):
        from services.query_classifier import QueryQualityAssessment

        quality = QueryQualityAssessment(
            is_unclear=False,
            reason=None,
            normalized_query="which class am i in",
            clarification_prompt=None,
        )
        node = AnalyzeQueryNode(_FakeClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "which class am i in",
                "conversation_history": [],
                "context_gathering_attempts": 0,
                "user_session_id": "memory-002",
            }
        )

        self.assertEqual(updates["query_type"], "conversational")
        self.assertEqual(updates["memory_intent"], "class_level")
        self.assertEqual(updates["query_en"], "which class am i in")

    async def test_analyze_query_locks_english_preference_from_instruction(self):
        from services.query_classifier import QueryQualityAssessment

        quality = QueryQualityAssessment(
            is_unclear=False,
            reason=None,
            normalized_query="always reply in english",
            clarification_prompt=None,
        )
        node = AnalyzeQueryNode(_FakeClassifier(quality), _FakeDetector())
        updates = await node(
            {
                "query": "always reply in english",
                "conversation_history": [],
                "context_gathering_attempts": 0,
                "session_metadata": {},
            }
        )
        self.assertEqual(updates["preferred_response_language"], "en")
        self.assertTrue(updates["language_preference_locked"])
        self.assertEqual(updates["language"], "en")

    async def test_analyze_query_does_not_use_cached_followup_for_class_noun(self):
        from services.query_classifier import QueryQualityAssessment
        from unittest.mock import AsyncMock

        quality = QueryQualityAssessment(
            is_unclear=False,
            reason=None,
            normalized_query="class?",
            clarification_prompt=None,
        )
        fake_classifier = _FakeClassifier(quality)
        node = AnalyzeQueryNode(fake_classifier, _FakeDetector())

        with patch("services.cache_service.CacheService.get", new=AsyncMock(return_value="photosynthesis basics")):
            updates = await node(
                {
                    "query": "class?",
                    "conversation_history": [],
                    "context_gathering_attempts": 0,
                    "user_session_id": "memory-003",
                }
            )

        self.assertEqual(updates["query_type"], "curriculum_specific")
        self.assertEqual(updates["query_en"], "class?")
        self.assertNotIn("Follow-up", updates["query_en"])

    async def test_low_rag_keeps_reactive_web_fallback_with_reason(self):
        node = RetrieveDocumentsNode(_FakeRetriever([]))
        state = {
            "query_en": "what is llm",
            "query": "what is llm",
            "request_filters": {},
            "intent": "concept_explanation",
        }
        with patch("tools.web_search_tool.WebSearchTool", _FakeWebSearchTool):
            updates = await node(state)

        self.assertEqual(updates["rag_quality"], "low")
        self.assertEqual(updates["web_fallback_reason"], "low_rag")
        self.assertEqual(updates["llm_calls"], 1)

    async def test_recency_reason_marks_speculative_skip(self):
        node = RetrieveDocumentsNode(_FakeRetriever([]))
        state = {
            "query_en": "latest updates in physics",
            "query": "latest updates in physics",
            "request_filters": {},
            "intent": "concept_explanation",
            "documents": [
                {
                    "id": "d1",
                    "score": 0.91,
                    "metadata": {"subject": "Physics"},
                    "text": "Recent physics class summary.",
                }
            ],
        }
        updates = await node(state)
        self.assertEqual(updates["web_fallback_reason"], "recency_skipped_speculative")
        self.assertEqual(updates["llm_calls"], 0)


class QueryQualityHeuristicsTests(unittest.TestCase):
    def test_query_quality_detects_gibberish(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "????")
        self.assertTrue(result.is_unclear)
        self.assertEqual(result.reason, "noisy_or_gibberish_query")

    def test_query_quality_expands_abbreviation(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "phy chapter 2")
        self.assertIn("physics", result.normalized_query)

    def test_query_quality_preserves_non_abbreviation_casing(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "Please explain PHY Chapter 2")
        self.assertIn("Please explain physics Chapter 2", result.normalized_query)

    def test_query_quality_does_not_expand_ambiguous_single_letter_q(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "q about gravity")
        self.assertNotIn("question", result.normalized_query)
        self.assertIn("q", result.normalized_query)

    def test_query_quality_expands_eng_only_with_educational_context(self):
        classifier = QueryClassifier.__new__(QueryClassifier)

        no_context = QueryClassifier.assess_query_quality(classifier, "eng pls")
        self.assertIn("eng", no_context.normalized_query)
        self.assertNotIn("english", no_context.normalized_query)

        with_context = QueryClassifier.assess_query_quality(classifier, "eng homework pls")
        self.assertIn("english homework", with_context.normalized_query)

    def test_query_quality_subject_detection_ignores_embedded_substring(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "bio pythonic pls")
        self.assertEqual(result.reason, "ambiguous_abbreviation_query")

    def test_query_quality_subject_detection_handles_trimmed_keyword_tokens(self):
        classifier = QueryClassifier.__new__(QueryClassifier)
        result = QueryClassifier.assess_query_quality(classifier, "bio python pls")
        self.assertFalse(result.is_unclear)
        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()
