"""Keep the existing terminal entry point usable after sharing the service."""

import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import MagicMock, patch

from google.genai.errors import ServerError
from src.rag_study_buddy import agent
from src.rag_study_buddy.rag import Evidence, Settings


class TerminalTests(unittest.TestCase):
    def test_terminal_uses_source_urls_instead_of_browser_anchors(self):
        session = MagicMock()
        session.stream_answer.return_value = iter([{
            "type": "complete", "text": "Supported answer [1](#evidence-test-1)",
            "evidence": [Evidence(1, "https://example.com/docs", 3, "Evidence text", "evidence-test-1")],
        }])
        output = io.StringIO()
        with patch.object(agent, "load_settings", return_value=Settings("offline-test-key")), patch.object(
            agent.RagSession, "prepare", return_value=session,
        ), redirect_stdout(output):
            self.assertEqual(agent.main(), 0)
        self.assertIn("Supported answer [1](https://example.com/docs)", output.getvalue())
        self.assertIn("chunk 3", output.getvalue())

    def test_api_failure_returns_nonzero_with_safe_message(self):
        output = io.StringIO()
        with patch.object(agent, "load_settings", return_value=Settings("offline-test-key")), patch.object(
            agent.RagSession, "prepare", side_effect=ServerError(503, {"error": {"message": "secret-detail"}}),
        ), redirect_stderr(output):
            self.assertEqual(agent.main(), 1)
        self.assertIn("overloaded", output.getvalue())
        self.assertNotIn("secret-detail", output.getvalue())


if __name__ == "__main__":
    unittest.main()
