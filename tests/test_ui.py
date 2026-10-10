"""Exercise the UI without downloading models or making paid API calls."""
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class AgricultureUITest(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.backend = ModuleType("rag")
        self.backend.retrieve = Mock(return_value=[{
            "source": "01_soil_health_management.pdf", "page": 2,
            "text": "Organic matter supports healthy soil.", "distance": 0.42,
        }])
        self.backend.generate_answer = Mock(return_value="Add organic matter to support soil health.")
        self.backend.prepare_query = Mock(return_value="improve soil health")
        self.backend.index_uploads = Mock()
        self.modules = patch.dict(sys.modules, {"rag": self.backend})
        self.modules.start()
        self.app = AppTest.from_file(APP).run()
        self.app.session_state.messages = []
        self.app.text_input(key="welcome_name").set_value("Manogna")
        self.app.button[0].click().run()

    def tearDown(self):
        self.modules.stop()

    def ask(self, text="How can I improve soil health?"):
        self.app.chat_input[0].set_value(text).run()
        self.assertFalse(self.app.exception)

    def test_starter_history_export_and_clear(self):
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_message), 0)
        self.app.button(key="starter_0").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertTrue(self.app.session_state.messages[-1]["content"].startswith("Manogna,"))
        self.assertEqual(self.app.session_state.messages[0]["content"], "How can farmers improve soil health?")
        self.assertIn("Sources · 1 document pages", [e.label for e in self.app.expander])
        self.assertEqual(len(self.app.get("download_button")), 1)
        self.app.run()
        self.assertEqual(len(self.app.chat_message), 2)
        self.ask("What about irrigation for rice?")
        self.assertEqual(len(self.app.chat_message), 4)
        next(b for b in self.app.button if "New conversation" in b.label).click().run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertEqual(len(self.app.get("download_button")), 0)
        self.assertEqual(self.app.session_state.user_name, "Manogna")

    def test_home_validates_name_and_resets_conversation(self):
        self.ask()
        next(b for b in self.app.button if b.label == "Change name / Home").click().run()
        self.assertEqual(self.app.session_state.user_name, "")
        self.assertEqual(len(self.app.chat_input), 0)
        self.assertEqual(self.app.session_state.messages, [])
        self.app.text_input(key="welcome_name").set_value("   ")
        self.app.button[0].click().run()
        self.assertEqual(len(self.app.warning), 1)
        self.app.text_input(key="welcome_name").set_value("  Asha   Rao  ")
        self.app.button[0].click().run()
        self.assertEqual(self.app.session_state.user_name, "Asha Rao")
        self.assertEqual(len(self.app.chat_input), 1)

    def test_failure_is_persisted_and_retry_succeeds(self):
        self.backend.generate_answer.side_effect = RuntimeError("Simulated provider failure")
        self.ask()
        self.assertTrue(self.app.session_state.messages[-1]["error"])
        self.backend.generate_answer.side_effect = None
        self.ask()
        self.assertIn("sources", self.app.session_state.messages[-1])

    def test_blank_question_makes_no_request(self):
        self.ask("   ")
        self.assertEqual(len(self.app.warning), 1)
        self.backend.retrieve.assert_not_called()

    def test_empty_retrieval_skips_generation(self):
        self.backend.retrieve.return_value = []
        self.ask()
        self.backend.generate_answer.assert_not_called()
        self.assertIn("couldn't find relevant passages", self.app.session_state.messages[-1]["content"])

    def test_followup_uses_history_and_selected_language(self):
        self.ask("How can I improve soil health?")
        self.app.selectbox(key="answer_language").select("తెలుగు · Telugu").run()
        self.ask("What about rice?")
        kwargs = self.backend.prepare_query.call_args.kwargs
        self.assertEqual(len(kwargs["history"]), 2)
        self.assertEqual(kwargs["history"][0]["content"], "How can I improve soil health?")
        self.assertEqual(self.backend.generate_answer.call_args.kwargs["language"], "Telugu")
        self.assertIn("వ్యవసాయ", self.app.session_state.messages[-1]["content"])

    def test_upload_scope_requires_index(self):
        self.app.selectbox(key="search_scope").select("Uploaded PDFs only").run()
        self.ask()
        self.assertEqual(len(self.app.warning), 1)
        self.backend.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
