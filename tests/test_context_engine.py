import os
import tempfile
import unittest
from unittest.mock import patch

import context
import database
from memory_importance import score_memory
from semantic_memory import index_message, retrieve


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

    def test_usage_stats_track_prompt_cache_tokens(self):
        database.record_llm_usage(
            "s1",
            "chat",
            "x-ai/grok-4-fast",
            1000,
            100,
            cached_input_tokens=600,
            cache_write_tokens=50,
        )

        stats = database.get_session_stats("s1")

        self.assertEqual(stats["cached_input_tokens"], 600)
        self.assertEqual(stats["cache_write_tokens"], 50)
        self.assertLess(stats["input_cost_usd"], database.calculate_cost("x-ai/grok-4-fast", 1000, 0)["input"])

    def test_retrieval_ranks_exact_memory_fact(self):
        fact_id = database.store_message_with_usage(
            "s1",
            "user",
            "Stable memory: the user's favorite tea is jasmine.",
        )
        index_message("s1", fact_id, "user", "Stable memory: the user's favorite tea is jasmine.")
        other_id = database.store_message_with_usage(
            "s1",
            "assistant",
            "Routine implementation chatter about formatting.",
        )
        index_message("s1", other_id, "assistant", "Routine implementation chatter about formatting.")

        results = retrieve("s1", "What is the user's favorite tea?", top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertIn("jasmine", results[0]["content"])
        self.assertGreater(results[0]["lexical_score"], 0)
        self.assertEqual(results[0]["embedding_model"], "mock/hash")
        hierarchy = database.get_memory_hierarchy("s1")
        self.assertGreaterEqual(hierarchy["top_memories"][0]["retrieval_count"], 1)

    def test_retrieval_modes_are_explicit(self):
        fact_id = database.store_message_with_usage(
            "s1",
            "user",
            "Project memory: Atlas launches in Europe.",
        )
        index_message("s1", fact_id, "user", "Project memory: Atlas launches in Europe.")
        other_id = database.store_message_with_usage(
            "s1",
            "assistant",
            "Routine implementation chatter about formatting.",
        )
        index_message("s1", other_id, "assistant", "Routine implementation chatter about formatting.")

        bm25 = retrieve("s1", "Where does Atlas launch?", top_k=1, mode="bm25")
        embedding = retrieve("s1", "Where does Atlas launch?", top_k=1, mode="embedding")

        self.assertEqual(bm25[0]["retrieval_mode"], "bm25")
        self.assertEqual(embedding[0]["retrieval_mode"], "embedding")
        self.assertEqual(bm25[0]["embedding_model"], "mock/hash")

    def test_adaptive_context_uses_retrieval_for_memory_question(self):
        for index in range(20):
            if index == 1:
                content = "Stable memory: the user's favorite tea is jasmine."
            else:
                content = f"message-{index}"
            role = "user" if index % 2 == 0 else "assistant"
            message_id = database.store_message_with_usage("s1", role, content)
            index_message("s1", message_id, role, content)

        built = context.build_context(
            "s1",
            model="mock/echo",
            query="What is the user's favorite tea?",
            policy="adaptive",
        )

        self.assertIn("Retrieved relevant prior facts", built[0]["content"])
        self.assertIn("jasmine", built[0]["content"])

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

    def test_memory_importance_scores_preferences_above_routine_chatter(self):
        preference = score_memory(
            "Durable user preference: the user's favorite tea is jasmine.",
            role="user",
            message_id=10,
            latest_message_id=12,
            retrieval_count=2,
        )
        routine = score_memory(
            "Routine turn: neutral implementation chatter about formatting.",
            role="assistant",
            message_id=1,
            latest_message_id=1000,
        )
        stale = score_memory(
            "Old stale memory: the previous editor used to be Vim.",
            role="user",
            message_id=1,
            latest_message_id=1000,
        )

        self.assertGreater(preference["importance_score"], routine["importance_score"])
        self.assertEqual(preference["memory_action"], "preserve")
        self.assertEqual(routine["memory_action"], "evict")
        self.assertNotEqual(stale["memory_action"], "preserve")


if __name__ == "__main__":
    unittest.main()
