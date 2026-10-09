"""Exercise Streamlit chat, evidence selection and failures without Gemini calls."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.rag_study_buddy.rag import Evidence, Settings, SOURCE_PAGES


class OfflineSession:
    chunk_count = 12

    def __init__(self):
        self.calls = []
        self.fail = False

    def stream_answer(self, question, history, *, turn_id):
        self.calls.append((question, history))
        evidence = [Evidence(1, SOURCE_PAGES["Subagents"], len(self.calls), "Supporting source text.", f"evidence-{turn_id}-1")]
        yield {"type": "status", "text": "Reading evidence…"}
        yield {"type": "evidence", "evidence": evidence}
        yield {"type": "draft", "text": "Partial answer"}
        if self.fail:
            raise RuntimeError("Do not show this secret SDK detail")
        yield {"type": "complete", "text": f"Supported answer [1](#evidence-{turn_id}-1)", "evidence": evidence}


class UiTests(unittest.TestCase):
    def setUp(self):
        self.session = OfflineSession()
        self.key = "offline-test-key"
        self.prepare = patch("src.rag_study_buddy.rag.RagSession.prepare", return_value=self.session).start()
        self.addCleanup(patch.stopall)

        def settings(api_key="", source_urls=None):
            return Settings(api_key or self.key, source_urls=tuple(source_urls))

        patch("src.rag_study_buddy.rag.load_settings", side_effect=settings).start()
        self.app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60,
        ).run()

    def assert_no_errors(self):
        self.assertEqual([error.value for error in self.app.exception], [])

    def ask(self, question):
        self.app.chat_input[0].set_value(question).run()
        self.assert_no_errors()

    def test_initial_render_has_no_network_and_missing_key_disables_input(self):
        self.assert_no_errors()
        self.prepare.assert_not_called()
        self.assertEqual(len(self.app.chat_input), 1)
        self.key = ""
        self.app.run()
        self.assertTrue(self.app.chat_input[0].disabled)
        self.assertTrue(self.app.button(key="prepare").disabled)

    def test_chat_reuses_session_and_selects_evidence_for_each_turn(self):
        self.ask("What is a subagent?")
        self.ask("When should I use it?")
        self.assertEqual(self.prepare.call_count, 1)
        self.assertEqual(len(self.app.chat_message), 0)
        messages = [item.value for item in self.app.markdown]
        self.assertIn("**Question:** What is a subagent?", messages)
        self.assertIn("**Question:** When should I use it?", messages)
        self.assertEqual(sum(value.startswith("**Answer:** ") for value in messages), 2)
        self.assertEqual(len(self.session.calls[1][1]), 2)
        turns = self.app.session_state["turns"]
        self.assertEqual(self.app.selectbox(key="evidence_turn").value, turns[-1]["id"])
        self.app.button(key=f"view-{turns[0]['id']}").click().run()
        self.assert_no_errors()
        self.assertEqual(self.app.selectbox(key="evidence_turn").value, turns[0]["id"])
        self.assertTrue(any(item.value == "Chunk 1" for item in self.app.caption))

    def test_new_conversation_retains_index_and_document_change_resets_session(self):
        self.ask("First question")
        self.app.button(key="clear").click().run()
        self.assert_no_errors()
        self.assertEqual(self.app.session_state["turns"], [])
        self.assertIs(self.app.session_state["rag_session"], self.session)
        self.app.multiselect(key="documents").set_value(["Subagents"]).run()
        self.assert_no_errors()
        self.assertIsNone(self.app.session_state["rag_session"])
        self.app.multiselect(key="documents").set_value([]).run()
        self.assertTrue(self.app.chat_input[0].disabled)

    def test_generation_failure_retains_evidence_and_allows_retry(self):
        self.session.fail = True
        self.ask("Question that fails")
        self.assertTrue(self.app.error)
        self.assertNotIn("secret SDK detail", self.app.error[0].value)
        self.assertEqual(len(self.app.session_state["turns"][0]["evidence"]), 1)
        self.session.fail = False
        self.ask("Try again")
        self.assertEqual(self.session.calls[-1][1], [])
        self.assertNotIn("error", self.app.session_state["turns"][-1])

    def test_preparation_failure_is_actionable_and_retry_works(self):
        self.prepare.side_effect = RuntimeError("secret SDK detail")
        self.app.button(key="prepare").click().run()
        self.assert_no_errors()
        self.assertTrue(self.app.error)
        self.assertIsNone(self.app.session_state["rag_session"])
        self.prepare.side_effect = None
        self.app.button(key="prepare").click().run()
        self.assert_no_errors()
        self.assertIs(self.app.session_state["rag_session"], self.session)


if __name__ == "__main__":
    unittest.main()
