"""Offline regression checks for the shared index and evidence stream."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from ollama import ResponseError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from src.rag_study_buddy import rag


class LocalEmbeddings(Embeddings):
    calls = []
    fail_on_call = None

    def __init__(self, model, **kwargs):
        self.model = model

    def embed_documents(self, texts):
        self.calls.append(len(texts))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("Simulated embedding failure")
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class OfflineAgentModel(BaseChatModel):
    """Fake provider; real task dispatch, subgraph and file access still execute."""
    mode: str = "ok"
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-structured-model"

    def with_structured_output(self, schema, **kwargs):
        if isinstance(schema, dict):
            if "quote_numbers" in schema["properties"]:
                assert schema["properties"]["quote_numbers"]["items"]["enum"]
            else:
                assert schema["$defs"]["ClaimSupport"]["properties"]["quote_id"]["enum"]
            return self | RunnableLambda(lambda message: json.loads(message.content))
        return self | RunnableLambda(lambda message: schema.model_validate_json(message.content))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = json.loads(messages[-1].content)
        analyst = "You are evidence-analyst" in messages[0].content
        self.calls.append({"analyst": analyst, "messages": messages, "payload": payload})
        quote = "A subagent handles an isolated task."
        if analyst:
            quotes = [item["number"] for item in payload["passage_spans"] if item["text"] == quote]
            if self.mode == "bad_quote":
                quotes = [999]
            elif self.mode == "no_quotes":
                quotes = []
            output = {"quote_numbers": quotes}
        else:
            output = {"claims": [{"text": "A subagent handles an isolated task.",
                                  "supports": [{"quote_id": "e1q1"}]}], "insufficient": False}
            if self.mode == "unknown_id":
                output["claims"][0]["supports"][0]["quote_id"] = "e999q1"
            elif self.mode == "wrong_support":
                output["claims"][0]["supports"][0]["quote_id"] = "e1q999"
            elif self.mode == "no_support":
                output["claims"][0]["supports"] = []
            elif self.mode == "forged_link":
                output["claims"][0]["text"] += " [9](https://fake.example)"
            elif self.mode == "no_claims":
                output["claims"] = []
        content = "invalid JSON" if self.mode == "invalid_json" else json.dumps(output)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class IndexTests(unittest.TestCase):
    def setUp(self):
        temporary_root = rag.PROJECT_ROOT / "work" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        LocalEmbeddings.calls = []
        LocalEmbeddings.fail_on_call = None
        self.document = "\n\n".join(f"Section {i}: " + "x" * 900 for i in range(85))
        self.download = MagicMock()
        self.download.text = self.document
        self.settings = rag.Settings(source_urls=("https://example.com/docs",))
        self.patches = [
            patch.object(rag.requests, "get", return_value=self.download),
            patch.object(rag, "OllamaEmbeddings", LocalEmbeddings),
            patch.object(rag, "check_ollama", return_value="embedding-digest-a"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def build(self):
        return rag.build_index(self.settings, lambda text: None, cache_dir=self.cache)

    def test_cache_reuse_and_content_invalidation(self):
        store, count = self.build()
        self.assertEqual(len(store.store), count)
        self.assertTrue(all(size <= 80 for size in LocalEmbeddings.calls))
        call_count = len(LocalEmbeddings.calls)
        reused, reused_count = self.build()
        self.assertEqual(len(LocalEmbeddings.calls), call_count)
        self.assertEqual(len(reused.store), reused_count)
        for item in reused.store.values():
            self.assertEqual(item["metadata"]["source"], self.settings.source_urls[0])
            self.assertIn("chunk_id", item["metadata"])
        self.download.text += "\n\nNew content changes the cache identity."
        self.build()
        self.assertGreater(len(LocalEmbeddings.calls), call_count)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

    def test_resume_after_second_batch_fails(self):
        LocalEmbeddings.fail_on_call = 2
        with self.assertRaises(RuntimeError):
            self.build()
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)
        LocalEmbeddings.fail_on_call = None
        store, count = self.build()
        self.assertEqual(len(store.store), count)
        self.assertEqual(LocalEmbeddings.calls[0], 80)
        self.assertLess(LocalEmbeddings.calls[-1], 80)

    def test_empty_documents_fail_before_network(self):
        with self.assertRaises(ValueError):
            rag.build_index(rag.Settings(source_urls=()), lambda text: None, cache_dir=self.cache)
        rag.requests.get.assert_not_called()
        rag.check_ollama.assert_not_called()

    def test_unavailable_ollama_fails_before_document_download(self):
        rag.check_ollama.side_effect = ConnectionError("secret-connection-detail")
        with self.assertRaises(ConnectionError):
            self.build()
        rag.requests.get.assert_not_called()

    def test_embedding_changes_invalidate_cache_but_chat_changes_do_not(self):
        self.build()
        count = len(LocalEmbeddings.calls)
        self.settings = replace(self.settings, chat_model="another-chat-model")
        self.build()
        self.assertEqual(len(LocalEmbeddings.calls), count)
        rag.check_ollama.return_value = "embedding-digest-b"
        self.build()
        self.assertGreater(len(LocalEmbeddings.calls), count)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)
        count = len(LocalEmbeddings.calls)
        self.settings = replace(self.settings, embedding_model="another-embedding-model")
        self.build()
        self.assertGreater(len(LocalEmbeddings.calls), count)


class OllamaTests(unittest.TestCase):
    def test_settings_use_local_defaults_without_gemini_credentials(self):
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_API_KEY": "unused"}, clear=True):
            settings = rag.load_settings()
        self.assertEqual(settings.base_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.embedding_model, rag.DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(settings.num_ctx, 8192)
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"OLLAMA_NUM_CTX": "0"}):
            with self.assertRaises(rag.OllamaSetupError):
                rag.load_settings()

    def test_chat_configuration_does_not_contact_server(self):
        settings = rag.Settings(base_url="http://127.0.0.1:11435", num_ctx=4096, timeout_seconds=240)
        with patch.object(rag, "ChatOllama") as chat:
            session = rag.RagSession(settings, MagicMock(), 12)
        self.assertIs(session.model, chat.return_value)
        self.assertEqual(chat.call_args.kwargs["model"], settings.chat_model)
        self.assertEqual(chat.call_args.kwargs["base_url"], settings.base_url)
        self.assertEqual(chat.call_args.kwargs["num_ctx"], 4096)
        self.assertIs(chat.call_args.kwargs["reasoning"], False)
        self.assertEqual(chat.call_args.kwargs["num_predict"], 1024)
        self.assertEqual(chat.call_args.kwargs["client_kwargs"]["timeout"], 240)

    def test_preflight_requires_downloaded_completion_and_embedding_models(self):
        settings = rag.Settings()
        with patch.object(rag, "Client") as client:
            api = client.return_value
            api.list.return_value.models = [
                SimpleNamespace(model=settings.chat_model, digest="chat-digest"),
                SimpleNamespace(model=settings.embedding_model, digest="embedding-digest"),
            ]
            api.show.side_effect = [SimpleNamespace(capabilities=["completion", "tools"]),
                                    SimpleNamespace(capabilities=["embedding"])]
            self.assertEqual(rag.check_ollama(settings), "embedding-digest")
            api.show.side_effect = None
            api.show.return_value.capabilities = ["embedding"]
            with self.assertRaisesRegex(rag.OllamaSetupError, "text generation"):
                rag.check_ollama(settings)
            api.list.return_value.models = []
            with self.assertRaisesRegex(rag.OllamaSetupError, "ollama pull"):
                rag.check_ollama(settings)
        with patch.object(rag, "Client") as client:
            with self.assertRaisesRegex(rag.OllamaSetupError, "cloud"):
                rag.check_ollama(replace(settings, chat_model="example:cloud"))
            client.assert_not_called()


class ToolRecorder(BaseCallbackHandler):
    def __init__(self):
        self.tools = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tools.append((serialized["name"], input_str, kwargs.get("metadata", {})))


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.document = Document(page_content="A subagent handles an isolated task.",
                                 metadata={"source": "https://example.com/subagents", "chunk_id": 7})
        self.store.similarity_search.return_value = [self.document]
        self.session = rag.RagSession.__new__(rag.RagSession)
        self.session.model = OfflineAgentModel()
        self.session.vector_store = self.store

    def test_real_deepagents_task_reads_each_file_before_analysis(self):
        self.store.similarity_search.return_value = [self.document, self.document, Document(
            page_content="Other information.", metadata={"source": "https://example.com/other", "chunk_id": 8})]
        recorder = ToolRecorder()
        history = [{"role": "user", "content": "What is a subagent?"},
                   {"role": "assistant", "content": "Previous answer is not evidence."}]
        events = list(self.session.stream_answer("When should I use it?", history,
                                                turn_id="turn-a", config={"callbacks": [recorder]}))
        final = events[-1]
        self.assertEqual(final["type"], "complete")
        self.assertEqual(len(final["evidence"]), 2)
        self.assertEqual(final["evidence"][0].chunk_id, 7)
        self.assertEqual(final["evidence"][0].text, self.document.page_content)
        self.assertEqual(final["text"], "A subagent handles an isolated task. [1](#evidence-turn-a-1)")
        self.assertEqual([item[0] for item in recorder.tools], ["search_documents", "task", "read_file", "task", "read_file"])
        for name, inputs, metadata in recorder.tools:
            if name == "task":
                self.assertIn("evidence-analyst", inputs)
            if name == "read_file":
                self.assertTrue(metadata.get("langgraph_checkpoint_ns"))
        calls = self.session.model.calls
        self.assertEqual([call["analyst"] for call in calls], [True, True, False])
        self.assertEqual(calls[0]["payload"]["passage_spans"][0]["text"], self.document.page_content)
        self.assertEqual(calls[1]["payload"]["passage_spans"][0]["text"], "Other information.")
        self.assertNotIn("Previous answer", str(calls[0]["payload"]))
        self.assertIn("Previous answer", str(calls[-1]["messages"]))
        self.assertIn("What is a subagent?", self.store.similarity_search.call_args.args[0])
        self.assertTrue(any(event["type"] == "evidence" for event in events))
        self.assertFalse(any(event["type"] == "draft" for event in events))
        self.assertNotIn("quotes", str(events))

    def test_invalid_quotes_and_citations_never_complete_or_publish_drafts(self):
        for mode in ("bad_quote", "unknown_id", "wrong_support", "no_support", "forged_link", "no_claims", "invalid_json"):
            with self.subTest(mode=mode):
                self.session.model = OfflineAgentModel(mode=mode)
                events = []
                with self.assertRaises(rag.RagFlowError):
                    for event in self.session.stream_answer("What is a subagent?"):
                        events.append(event)
                self.assertTrue(any(event["type"] == "evidence" for event in events))
                self.assertFalse(any(event["type"] in ("complete", "draft") for event in events))

    def test_no_evidence_or_relevance_returns_explicit_abstention(self):
        self.store.similarity_search.return_value = []
        final = list(self.session.stream_answer("Missing topic?"))[-1]
        self.assertEqual(final["text"], rag.INSUFFICIENT_ANSWER)
        self.assertEqual(final["evidence"], [])
        self.assertEqual(self.session.model.calls, [])
        self.store.similarity_search.return_value = [self.document]
        self.session.model = OfflineAgentModel(mode="no_quotes")
        final = list(self.session.stream_answer("Missing topic?"))[-1]
        self.assertEqual(final["text"], rag.INSUFFICIENT_ANSWER)
        self.assertEqual(len(self.session.model.calls), 1)

    def test_analysis_failure_stops_synthesis(self):
        self.session.model = OfflineAgentModel(mode="bad_quote")
        with self.assertRaises(rag.RagFlowError):
            list(self.session.stream_answer("Explain subagents"))
        self.assertEqual(len(self.session.model.calls), 1)
        self.assertTrue(self.session.model.calls[0]["analyst"])

    def test_unreadable_evidence_stops_before_model_generation(self):
        from deepagents.backends.protocol import FileDownloadResponse
        with patch.object(rag.StateBackend, "download_files", return_value=[
            FileDownloadResponse(path="/missing", content=None, error="file_not_found"),
        ]):
            with self.assertRaisesRegex(rag.RagFlowError, "could not be read"):
                list(self.session.stream_answer("What is a subagent?"))
        self.assertEqual(self.session.model.calls, [])

    def test_turns_never_reuse_old_citation_anchors(self):
        first = list(self.session.stream_answer("What is a subagent?", turn_id="one"))[-1]
        second = list(self.session.stream_answer("What is a subagent?", turn_id="two"))[-1]
        self.assertIn("#evidence-one-1", first["text"])
        self.assertIn("#evidence-two-1", second["text"])
        self.assertNotIn("#evidence-one-1", second["text"])

    def test_error_messages_hide_sensitive_sdk_details(self):
        for error, expected in [
            (ResponseError("secret-sdk-detail", status_code=503), "busy"),
            (ResponseError("secret-sdk-detail", status_code=404), "not found"),
            (ResponseError("secret-sdk-detail", status_code=500), "RAM/VRAM"),
            (ConnectionError("secret-sdk-detail"), "Cannot connect"),
            (httpx.ReadTimeout("secret-sdk-detail"), "timed out"),
            (RuntimeError("secret-sdk-detail"), "could not be completed"),
        ]:
            wrapped = RuntimeError("outer")
            wrapped.__cause__ = error
            text = rag.describe_error(wrapped)
            self.assertIn(expected, text)
            self.assertNotIn("secret-sdk-detail", text)


if __name__ == "__main__":
    unittest.main()
