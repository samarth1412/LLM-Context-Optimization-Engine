import os
import tempfile
import unittest
from unittest.mock import patch

import context
import database


def fake_usage(input_tokens=100, output_tokens=20):
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated": False,
    }


class ContextEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_db = database.DB_NAME
        database.DB_NAME = os.path.join(self.tmpdir.name, "test.db")
        database.init_database()

    def tearDown(self):
        database.DB_NAME = self.original_db
        self.tmpdir.cleanup()

    def add_messages(self, session_id, count):
        for index in range(count):
            role = "user" if index % 2 == 0 else "assistant"
            database.store_message_with_usage(
                session_id,
                role,
                f"message-{index}",
            )

    def test_long_context_uses_summary_and_recent_messages(self):
        self.add_messages("s1", 18)

        with patch.object(
            context,
            "generate_summary",
            return_value=("summary for test", fake_usage()),
        ):
            built = context.build_context("s1", model="mock/echo")

        self.assertEqual(len(built), 16)
        self.assertEqual(built[0]["role"], "system")
        self.assertIn("summary for test", built[0]["content"])
        self.assertEqual(built[1]["content"], "message-3")
        self.assertEqual(built[-1]["content"], "message-17")

    def test_cached_summary_avoids_regeneration(self):
        self.add_messages("s1", 18)

        with patch.object(
            context,
            "generate_summary",
            return_value=("cached summary", fake_usage()),
        ):
            context.build_context("s1", model="mock/echo")

        with patch.object(context, "generate_summary", side_effect=AssertionError):
            built = context.build_context("s1", model="mock/echo")

        self.assertIn("cached summary", built[0]["content"])

    def test_last_n_messages_use_insert_order(self):
        self.add_messages("s1", 5)

        last_three = database.get_last_n_messages("s1", 3)

        self.assertEqual([msg["content"] for msg in last_three], ["message-2", "message-3", "message-4"])

    def test_usage_stats_include_background_llm_calls(self):
        database.record_llm_usage("s1", "chat", "x-ai/grok-4-fast", 1000, 2000)
        database.record_llm_usage("s1", "summary", "x-ai/grok-4-fast", 500, 100)

        stats = database.get_session_stats("s1")

        self.assertEqual(stats["chat_input_tokens"], 1000)
        self.assertEqual(stats["chat_output_tokens"], 2000)
        self.assertEqual(stats["background_input_tokens"], 500)
        self.assertEqual(stats["background_output_tokens"], 100)
        self.assertGreater(stats["total_cost_usd"], 0)

    def test_delete_session_removes_all_session_state(self):
        self.add_messages("s1", 3)
        database.cache_summary("s1", 1, "summary")
        database.save_story_context("s1", "original story")
        database.record_llm_usage("s1", "chat", "mock/echo", 10, 5)

        deleted = database.delete_session("s1")

        self.assertEqual(deleted, 3)
        self.assertEqual(database.count_messages("s1"), 0)
        self.assertIsNone(database.get_story_context("s1"))
        self.assertEqual(database.get_cached_summary("s1", 1), None)
        self.assertEqual(database.get_session_stats("s1")["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
