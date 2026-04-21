import unittest

from langchain_core.messages import HumanMessage

from agents.conversational_agent import ConversationalAgent


class _FakeLLM:
    async def ainvoke(self, _prompt, config=None):
        raise AssertionError("LLM should not be called for deterministic memory responses")


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


if __name__ == "__main__":
    unittest.main()
