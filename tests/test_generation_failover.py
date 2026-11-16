"""Exercise the generation priority and REST contracts with offline responses."""

import json
import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.documents import Document

from src.rag_study_buddy import gemini_rag as rag


def groq_answer(quote_id="e1q1"):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(cited_answer(quote_id))}}],
    })


def cited_answer(quote_id="e1q1"):
    return {"claims": [{"text": "Evidence text.", "supports": [{"quote_id": quote_id}]}], "insufficient": False}


def cloudflare_answer(quote_id="e1q1", *, as_text=False):
    answer = cited_answer(quote_id)
    return httpx.Response(200, json={
        "success": True, "errors": [], "messages": [],
        "result": {"response": json.dumps(answer) if as_text else answer},
    })


class FailoverTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(rag, "_LIMITERS", {}))
        self.clock = self.enterContext(patch.object(rag, "monotonic", return_value=100))
        self.requests = []
        self.store = MagicMock()
        self.store.similarity_search.return_value = [
            Document(page_content="Evidence text.", metadata={"source": "https://example.com", "chunk_id": 7}),
        ]
        self.settings = rag.Settings(
            api_key="offline-gemini", groq_api_key="offline-groq",
            cloudflare_api_token="offline-cloudflare", cloudflare_account_id="a" * 32,
        )

    def mock_http(self, *responses):
        replies = iter(responses)

        def handle(request):
            self.requests.append(request)
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client = self.enterContext(httpx.Client(transport=httpx.MockTransport(handle)))
        self.enterContext(patch("src.rag_study_buddy.generation.httpx.post", side_effect=client.post))
        return client

    def session(self, settings=None, transport=None):
        session = rag.RagSession(settings or self.settings, self.store, 1)
        if session.model is not None:
            self.addCleanup(session.model.client.close)
            if transport is not None:
                self.enterContext(patch.object(session.model.client._api_client, "_httpx_client", transport))
        return session

    def test_gemini_and_groq_429_route_to_cloudflare_with_one_retrieval(self):
        transport = self.mock_http(
            httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "Gemini quota"}}),
            httpx.Response(429, headers={"Retry-After": "90"}),
            cloudflare_answer(),
            cloudflare_answer(as_text=True),
        )
        session = self.session(transport=transport)
        events = list(session.stream_answer("Question?", turn_id="first"))
        self.store.similarity_search.assert_called_once_with("Question?", k=4)
        self.assertEqual(events[-1]["text"], "Evidence text. [1](#evidence-first-1)")
        self.assertEqual(events[-1]["provider"], "Cloudflare Workers AI")
        self.assertEqual(events[-1]["evidence"][0].chunk_id, 7)
        self.assertEqual([request.url.host for request in self.requests], [
            "generativelanguage.googleapis.com", "api.groq.com", "api.cloudflare.com",
        ])
        _, groq, cloudflare = self.requests
        self.assertEqual(groq.headers["Authorization"], "Bearer offline-groq")
        self.assertEqual(cloudflare.headers["Authorization"], "Bearer offline-cloudflare")
        self.assertEqual(groq.url.path, "/openai/v1/chat/completions")
        account_path = f"/client/v4/accounts/{self.settings.cloudflare_account_id}"
        self.assertEqual(cloudflare.url.path, f"{account_path}/ai/run/{self.settings.cloudflare_chat_model}")
        groq_payload, cloudflare_payload = json.loads(groq.content), json.loads(cloudflare.content)
        self.assertTrue(groq_payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(groq_payload["messages"], cloudflare_payload["messages"])
        self.assertEqual(cloudflare_payload["max_tokens"], 2048)
        self.assertEqual(cloudflare_payload["response_format"]["type"], "json_schema")
        schema = cloudflare_payload["response_format"]["json_schema"]
        self.assertEqual(schema, groq_payload["response_format"]["json_schema"]["schema"])
        quote_id = schema["properties"]["claims"]["items"]["properties"]["supports"]["items"]["properties"]
        self.assertEqual(quote_id, {"quote_id": {"type": "string"}})

        # Another session asking a different question skips both cooling providers.
        second = self.session(transport=transport)
        result = list(second.stream_answer("Another question", turn_id="second"))[-1]
        self.assertEqual(result["text"], "Evidence text. [1](#evidence-second-1)")
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.requests[-1].url.host, cloudflare.url.host)

    def test_gemini_and_groq_client_timeouts_route_to_cloudflare_with_one_retrieval(self):
        transport = self.mock_http(
            httpx.ReadTimeout("Gemini detail"), httpx.ConnectTimeout("Groq detail"), cloudflare_answer(),
        )
        events = list(self.session(transport=transport).stream_answer("Question?", turn_id="timeout"))
        self.store.similarity_search.assert_called_once_with("Question?", k=4)
        self.assertEqual(events[-1]["provider"], "Cloudflare Workers AI")
        self.assertEqual(events[-1]["text"], "Evidence text. [1](#evidence-timeout-1)")
        self.assertIn({"type": "status", "text": "Gemini timed out."}, events)
        self.assertIn({"type": "status", "text": "Groq timed out."}, events)
        self.assertEqual([request.url.host for request in self.requests], [
            "generativelanguage.googleapis.com", "api.groq.com", "api.cloudflare.com",
        ])

    def test_provider_timeout_status_codes_route_to_groq(self):
        transport = self.mock_http(
            httpx.Response(408, json={"success": False, "errors": [{"code": 3007}]}),
            httpx.Response(504, json={"error": {"status": "DEADLINE_EXCEEDED"}}),
            groq_answer(),
        )
        settings = replace(self.settings, provider_order=("cloudflare", "gemini", "groq"))
        events = list(self.session(settings, transport=transport).stream_answer("Question?"))
        self.assertEqual(events[-1]["provider"], "Groq")
        self.assertIn({"type": "status", "text": "Cloudflare Workers AI timed out."}, events)
        self.assertIn({"type": "status", "text": "Gemini timed out."}, events)
        self.assertEqual(len(self.requests), 3)

    def test_timeout_and_quota_exhaustion_report_safe_error_then_retry_priority_on_new_turn(self):
        self.mock_http(
            httpx.ReadTimeout("secret timeout detail"),
            httpx.Response(429, json={"success": False, "errors": [{"code": 3036}]}),
            groq_answer(),
        )
        session = self.session(replace(self.settings, api_key=""))
        with self.assertRaisesRegex(rag.RagFlowError, "The request timed out for Groq") as error:
            list(session.stream_answer("Question?"))
        self.assertNotIn("secret timeout detail", rag.describe_error(error.exception))
        result = list(session.stream_answer("Another question"))[-1]
        self.assertEqual(result["provider"], "Groq")
        self.assertEqual([request.url.host for request in self.requests], [
            "api.groq.com", "api.cloudflare.com", "api.groq.com",
        ])

    def test_local_gemini_allowance_overflow_goes_directly_to_groq(self):
        transport = self.mock_http(groq_answer())
        settings = replace(self.settings, requests_per_minute=1)
        rag._get_limiter(settings).acquire()
        result = list(self.session(settings, transport=transport).stream_answer("Question"))[-1]
        self.assertEqual(result["provider"], "Groq")
        self.assertEqual([request.url.host for request in self.requests], ["api.groq.com"])

    def test_providers_resume_in_order_when_their_retry_after_expires(self):
        self.mock_http(
            httpx.Response(429, headers={"Retry-After": "90"}, json={"message": "secret Groq detail"}),
            httpx.Response(429, headers={"Retry-After": "80"}, json={"message": "secret Cloudflare detail"}),
            groq_answer(),
        )
        session = self.session(replace(self.settings, api_key=""))
        with self.assertRaisesRegex(rag.RagFlowError, "^All configured generation providers are at their request limits"):
            list(session.stream_answer("Question"))
        self.clock.return_value = 170
        with self.assertRaises(rag.RagFlowError):
            list(session.stream_answer("Still cooling"))
        self.assertEqual(len(self.requests), 2)
        self.clock.return_value = 191
        result = list(session.stream_answer("After cooldown"))[-1]
        self.assertEqual(result["provider"], "Groq")
        self.assertEqual(len(self.requests), 3)

    def test_depleted_gemini_credits_use_groq(self):
        transport = self.mock_http(httpx.Response(402, json={"error": {"status": "RESOURCE_EXHAUSTED"}}), groq_answer())
        result = list(self.session(transport=transport).stream_answer("Question"))[-1]
        self.assertEqual(result["provider"], "Groq")
        self.assertEqual(len(self.requests), 2)

    def test_groq_configuration_errors_and_invalid_citations_stop_generation(self):
        self.mock_http(httpx.Response(401, json={"message": "secret detail"}), groq_answer("unknown-span"))
        session = self.session(replace(self.settings, provider_order=("groq", "cloudflare")))
        with self.assertRaisesRegex(rag.RagFlowError, "^Groq returned HTTP 401"):
            list(session.stream_answer("Question"))
        self.assertEqual(len(self.requests), 1)
        with self.assertRaisesRegex(rag.RagFlowError, "unverified evidence"):
            list(session.stream_answer("Question"))
        self.assertEqual([request.url.host for request in self.requests], ["api.groq.com", "api.groq.com"])

    def test_cloudflare_invalid_citations_stop_generation(self):
        self.mock_http(cloudflare_answer("unknown-span"))
        session = self.session(replace(self.settings, provider_order=("cloudflare", "groq")))
        with self.assertRaisesRegex(rag.RagFlowError, "unverified evidence"):
            list(session.stream_answer("Question"))
        self.assertEqual([request.url.host for request in self.requests], ["api.cloudflare.com"])

    def test_settings_enable_only_providers_in_order_with_keys_and_redact_keys(self):
        env = {"RAG_PROVIDER_ORDER": " cloudflare, groq ", "GEMINI_API_KEY": "secret-one",
               "GROQ_API_KEY": "secret-two", "CLOUDFLARE_API_TOKEN": "secret-three",
               "CLOUDFLARE_ACCOUNT_ID": "a" * 32}
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, env, clear=True):
            settings = rag.load_settings()
        self.assertEqual(list(settings.generation_models), ["cloudflare", "groq"])
        self.assertNotIn("secret-", repr(settings))
        self.assertIsNone(self.session(settings).model)

    def test_cloudflare_token_requires_valid_account_id(self):
        for account_id in ("", "invalid"):
            env = {"CLOUDFLARE_API_TOKEN": "secret-token", "CLOUDFLARE_ACCOUNT_ID": account_id}
            with self.subTest(account_id=account_id), patch.object(rag, "load_dotenv"), patch.dict(
                os.environ, env, clear=True,
            ):
                with self.assertRaisesRegex(rag.GeminiSetupError, "CLOUDFLARE_ACCOUNT_ID"):
                    rag.load_settings()

    def test_invalid_provider_order_is_reported(self):
        for order in ("gemini,gemini", "unknown", "groq,"):
            with self.subTest(order=order), patch.object(rag, "load_dotenv"), patch.dict(
                os.environ, {"RAG_PROVIDER_ORDER": order}, clear=True,
            ):
                with self.assertRaisesRegex(rag.GeminiSetupError, "RAG_PROVIDER_ORDER"):
                    rag.load_settings()
