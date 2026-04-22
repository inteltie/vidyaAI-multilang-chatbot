import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.conversational_agent import ConversationalAgent


class _FakeLLM:
    async def ainvoke(self, _prompt, config=None):
        raise AssertionError("LLM should not be called for deterministic memory responses")


class _FakeLLMInformational:
    async def ainvoke(self, _prompt, config=None):
        class _Resp:
            content = "India has seen mixed outcomes in growth, infrastructure, and policy debates."
            usage_metadata = {"input_tokens": 10, "output_tokens": 8}
        return _Resp()


class _FakeWebSearchTool:
    async def execute(self, query):
        return f"WEB_SEARCH_OBSERVATION for '{query}': short fact summary", 2, 3


class ConversationalMemoryReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_intent_name_uses_structured_summary(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "what is my name",
                "memory_intent": "name",
                "session_metadata": {
                    "structured_summary": {
                        "key_entities": {"name": "Harsh"},
                    }
                },
                "conversation_history": [],
            }
        )

        self.assertEqual(updates["response"], "Your name is Harsh.")
        self.assertEqual(updates["llm_calls"], 0)

    async def test_memory_intent_class_uses_recent_history(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "which class am i in",
                "memory_intent": "class_level",
                "session_metadata": {"structured_summary": {"key_entities": {}}},
                "conversation_history": [
                    HumanMessage(content="hi i am harsh from class 10 and i need help")
                ],
            }
        )

        self.assertEqual(updates["response"], "You mentioned you are in Class 10.")
        self.assertEqual(updates["llm_calls"], 0)

    async def test_memory_intent_name_missing_returns_clarification(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "what is my name",
                "memory_intent": "name",
                "session_metadata": {},
                "conversation_history": [],
            }
        )

        self.assertIn("do not have your name", updates["response"])
        self.assertEqual(updates["llm_calls"], 0)

    async def test_acknowledgment_only_turn_is_deterministic(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "ohkkk",
                "session_metadata": {},
                "conversation_history": [],
            }
        )
        self.assertIn("Got it", updates["response"])
        self.assertEqual(updates["llm_calls"], 0)

    async def test_existing_response_from_analyze_node_is_respected(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "My name is jay and Iam in class 12 th",
                "response": "Thanks, Jay. Noted that you are in Class 12. Please share your specific doubt and I will help.",
                "session_metadata": {},
                "conversation_history": [],
            }
        )
        self.assertIn("Thanks, Jay", updates["response"])
        self.assertEqual(updates["llm_calls"], 0)

    async def test_ongoing_greeting_does_not_repeat_hi_again(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "hi",
                "session_metadata": {},
                "conversation_history": [HumanMessage(content="previous turn")],
                "is_session_restart": False,
            }
        )
        self.assertEqual(updates["response"], "How can I help you with your question today?")
        self.assertEqual(updates["llm_calls"], 0)

    async def test_restart_turn_allows_single_reconnect_greeting(self):
        agent = ConversationalAgent(_FakeLLM())
        updates = await agent(
            {
                "query": "hello",
                "session_metadata": {},
                "conversation_history": [HumanMessage(content="previous turn")],
                "is_session_restart": True,
            }
        )
        self.assertIn("Welcome back!", updates["response"])
        self.assertEqual(updates["llm_calls"], 0)

    async def test_informational_conversational_turn_uses_web_path(self):
        agent = ConversationalAgent(_FakeLLMInformational())
        with patch("agents.conversational_agent.WebSearchTool", _FakeWebSearchTool):
            updates = await agent(
                {
                    "query": "what is current state of india under modi government",
                    "session_metadata": {},
                    "conversation_history": [HumanMessage(content="previous turn")],
                    "language": "en",
                }
            )
        self.assertIn("mixed outcomes", updates["response"])
        self.assertEqual(updates["llm_calls"], 1)
        self.assertGreater(updates["input_tokens"], 0)
        self.assertGreater(updates["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
