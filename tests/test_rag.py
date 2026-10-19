"""Offline regression checks for the shared index and evidence stream."""

import json
import os
import re
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
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

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
    """Exercise the real graph, delegation and file tools without a provider."""

    @property
    def _llm_type(self):
        return "offline-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        analyst = "You receive a question and the path" in messages[0].content
        if analyst and not isinstance(last, ToolMessage):
            path = re.search(r"/evidence/[^\s]+\.md", last.content).group()
            response = AIMessage(content="", tool_calls=[{
                "name": "read_file", "args": {"file_path": path}, "id": "read-1", "type": "tool_call",
            }])
        elif analyst:
            citation = re.search(r"\[1\]\(#[^)]+\)", last.content).group()
            response = AIMessage(content=f"ANALYST-ONLY: A subagent handles an isolated task. {citation}")
        elif not isinstance(last, ToolMessage):
            response = AIMessage(content="", tool_calls=[{
                "name": "search_documents", "args": {"query": "subagents"}, "id": "search-1", "type": "tool_call",
            }])
        elif last.name == "search_documents":
            path = json.loads(last.content)[0]["path"]
            response = AIMessage(content="", tool_calls=[{
                "name": "task", "args": {"subagent_type": "evidence-analyst", "description": f"What is a subagent? Read {path}"},
                "id": "task-1", "type": "tool_call",
            }])
        else:
            citation = re.search(r"\[1\]\(#[^)]+\)", last.content).group()
            response = AIMessage(content=f"A subagent handles an isolated task. {citation}")
        return ChatResult(generations=[ChatGeneration(message=response)])


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
        self.assertEqual(chat.call_args.kwargs["client_kwargs"]["timeout"], 240)

    def test_preflight_requires_downloaded_tool_and_embedding_models(self):
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
            api.show.return_value.capabilities = ["completion"]
            with self.assertRaisesRegex(rag.OllamaSetupError, "tool calling"):
                rag.check_ollama(settings)
            api.list.return_value.models = []
            with self.assertRaisesRegex(rag.OllamaSetupError, "ollama pull"):
                rag.check_ollama(settings)
        with patch.object(rag, "Client") as client:
            with self.assertRaisesRegex(rag.OllamaSetupError, "cloud"):
                rag.check_ollama(replace(settings, chat_model="example:cloud"))
            client.assert_not_called()


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.store.similarity_search.return_value = [
            Document(page_content="A subagent handles an isolated task.",
                     metadata={"source": "https://example.com/subagents", "chunk_id": 7}),
        ]
        self.session = rag.RagSession.__new__(rag.RagSession)
        self.session.model = MagicMock()
        self.session.vector_store = self.store

    def graph_factory(self, **kwargs):
        search = kwargs["tools"][0]
        graph = MagicMock()

        def stream(state, **options):
            self.received_history = state["messages"]
            manifest = json.loads(search.invoke({"query": "subagents"}))
            self.assertEqual(manifest[0]["chunk_id"], 7)
            self.assertEqual(manifest[0]["citation"], "[1](#evidence-turn-a-1)")
            # A repeated search must keep the same citation and deduplicate passages.
            search.invoke({"query": "isolated tasks"})
            self.assertEqual(options["version"], "v2")
            yield {"ns": ("tools:analyst",), "type": "messages", "data": (AIMessageChunk(content="private analysis"), {"langgraph_node": "model"})}
            yield {"ns": (), "type": "messages", "data": (AIMessageChunk(content="Tool preamble", id="tool-message"), {"langgraph_node": "model"})}
            yield {"ns": (), "type": "messages", "data": (AIMessageChunk(content="", id="tool-message", tool_call_chunks=[{
                "name": "search_documents", "args": "{}", "id": "call-1", "index": 0,
            }]), {"langgraph_node": "model"})}
            yield {"ns": (), "type": "messages", "data": (AIMessageChunk(content="A subagent ", id="answer"), {"langgraph_node": "model"})}
            yield {"ns": (), "type": "messages", "data": (AIMessageChunk(content="handles a task. [1](#evidence-turn-a-1)", id="answer"), {"langgraph_node": "model"})}
            yield {"ns": (), "type": "values", "data": {"messages": [
                AIMessage(content="", tool_calls=[{"name": "task", "args": {"subagent_type": "evidence-analyst"}, "id": "task-check"}]),
                AIMessage(content="A subagent handles a task. [1](#evidence-turn-a-1)")
            ]}}

        graph.stream.side_effect = stream
        return graph

    def test_actual_retrieved_evidence_and_only_coordinator_stream(self):
        with patch.object(rag, "StateBackend") as backend, patch.object(rag, "create_deep_agent", side_effect=self.graph_factory):
            events = list(self.session.stream_answer(
                "Explain it", [{"role": "user", "content": "What is a subagent?"}], turn_id="turn-a",
            ))
        final = events[-1]
        self.assertEqual(final["type"], "complete")
        self.assertEqual(len(final["evidence"]), 1)
        self.assertEqual(final["evidence"][0].chunk_id, 7)
        self.assertEqual(final["evidence"][0].text, self.store.similarity_search.return_value[0].page_content)
        self.assertEqual(self.received_history[-1]["content"], "Explain it")
        self.assertEqual(backend.return_value.upload_files.call_count, 2)
        self.assertNotIn("private analysis", str(events))
        self.assertIn({"type": "draft", "text": ""}, events)
        self.assertTrue(any(event["type"] == "evidence" for event in events))

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

    def test_model_cannot_complete_without_retrieval_or_delegation(self):
        for retrieved in (False, True):
            def factory(**kwargs):
                graph = MagicMock()
                def stream(*args, **options):
                    if retrieved:
                        kwargs["tools"][0].invoke({"query": "subagents"})
                    yield {"ns": (), "type": "values", "data": {"messages": [AIMessage(content="Unsupported answer")]}}
                graph.stream.side_effect = stream
                return graph
            with patch.object(rag, "StateBackend"), patch.object(rag, "create_deep_agent", side_effect=factory):
                with self.assertRaisesRegex(rag.RagFlowError, "skipped"):
                    list(self.session.stream_answer("What is a subagent?"))

    def test_empty_retrieval_can_report_insufficient_information(self):
        self.store.similarity_search.return_value = []
        def factory(**kwargs):
            graph = MagicMock()
            def stream(*args, **options):
                self.assertEqual(json.loads(kwargs["tools"][0].invoke({"query": "missing topic"})), [])
                yield {"ns": (), "type": "values", "data": {"messages": [AIMessage(content="Insufficient information.")]}}
            graph.stream.side_effect = stream
            return graph
        with patch.object(rag, "StateBackend"), patch.object(rag, "create_deep_agent", side_effect=factory):
            final = list(self.session.stream_answer("Missing topic?"))[-1]
        self.assertEqual(final["type"], "complete")
        self.assertEqual(final["evidence"], [])

    def test_real_graph_can_read_evidence_and_delegate_without_network(self):
        self.session.model = OfflineAgentModel()
        events = list(self.session.stream_answer("What is a subagent?", turn_id="integration"))
        self.assertEqual(events[-1]["type"], "complete")
        self.assertIn("[1](#evidence-integration-1)", events[-1]["text"])
        self.assertEqual(len(events[-1]["evidence"]), 1)
        self.assertNotIn("ANALYST-ONLY", str(events))


if __name__ == "__main__":
    unittest.main()
