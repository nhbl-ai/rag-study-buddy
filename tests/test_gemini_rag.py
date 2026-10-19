"""Offline contracts for single-request generation and citation provenance."""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from google.genai.errors import ClientError
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from src.rag_study_buddy import gemini_rag as rag


class OfflineModel(BaseChatModel):
    mode: str = "ok"
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-single-request"

    def with_structured_output(self, schema, **kwargs):
        assert kwargs['method'] == 'json_schema'
        assert schema['$defs']['ClaimSupport']['properties']['quote_id']['enum']
        return self | RunnableLambda(lambda message: json.loads(message.content))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = json.loads(messages[-1].content)
        self.calls.append({'messages': messages, 'payload': payload, 'kwargs': kwargs})
        output = {'claims': [{'text': 'A subagent handles an isolated task.',
                             'supports': [{'quote_id': payload['source_spans'][0]['quote_id']}]}],
                  'insufficient': False}
        if self.mode == 'unknown_id':
            output['claims'][0]['supports'][0]['quote_id'] = 'e999q1'
        elif self.mode == 'wrong_span':
            output['claims'][0]['supports'][0]['quote_id'] = 'e1q999'
        elif self.mode == 'no_support':
            output['claims'][0]['supports'] = []
        elif self.mode == 'forged_link':
            output['claims'][0]['text'] += ' [1](https://fake.example)'
        elif self.mode == 'blank_claim':
            output['claims'][0]['text'] = '   '
        elif self.mode == 'no_claims':
            output['claims'] = []
        elif self.mode == 'abstain':
            output.update(claims=[], insufficient=True)
        elif self.mode == 'duplicate_support':
            output['claims'][0]['supports'] *= 2
        content = 'invalid JSON' if self.mode == 'invalid_json' else json.dumps(output)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class GenerationTests(unittest.TestCase):
    def session(self, mode='ok', documents=None):
        documents = documents if documents is not None else [
            Document(page_content='A subagent handles an isolated task. Use it for specialized work.',
                     metadata={'source': f'https://example.com/{i}', 'chunk_id': i}) for i in range(4)
        ]
        store=MagicMock()
        store.similarity_search.return_value=documents
        model=OfflineModel(mode=mode)
        with patch.object(rag, 'ChatGoogleGenerativeAI', return_value=model):
            session=rag.RagSession(rag.Settings(api_key='offline',quota_group=uuid4().hex),store,len(documents))
        return session,model,store

    def test_four_passages_need_one_generation_and_code_builds_citations(self):
        session,model,store=self.session()
        events=list(session.stream_answer('What is a subagent?',turn_id='own-turn'))
        self.assertEqual(len(model.calls),1)
        self.assertEqual(model.calls[0]['kwargs']['automatic_function_calling'],{'disable':True})
        store.similarity_search.assert_called_once_with('What is a subagent?',k=4)
        self.assertEqual(len(events[-1]['evidence']),4)
        self.assertEqual(events[-1]['text'],'A subagent handles an isolated task. [1](#evidence-own-turn-1)')
        self.assertFalse(any(e['type']=='draft' for e in events))
        self.assertEqual({s['quote_id'] for s in model.calls[0]['payload']['source_spans']},
                         {f'e{i}q{j}' for i in range(1,5) for j in (1,2)})

    def test_invalid_output_never_publishes_or_retries(self):
        for mode in ('unknown_id','wrong_span','no_support','forged_link','blank_claim','invalid_json','no_claims'):
            with self.subTest(mode=mode):
                session,model,_=self.session(mode)
                events=[]
                with self.assertRaises(rag.RagFlowError):
                    for event in session.stream_answer('Question'):events.append(event)
                self.assertEqual(len(model.calls),1)
                self.assertFalse(any(e['type'] in ('complete','draft') for e in events))
                self.assertTrue(any(e['type']=='evidence' for e in events))

    def test_empty_retrieval_or_blank_passage_uses_no_gemini(self):
        for documents in ([],[Document(page_content='  \n ',metadata={'source':'https://example.com','chunk_id':1})]):
            with self.subTest(documents=documents):
                session,model,store=self.session(documents=documents)
                events=list(session.stream_answer('Question'))
                self.assertEqual(model.calls,[])
                self.assertEqual(events[-1]['text'],rag.INSUFFICIENT_ANSWER)
                self.assertEqual(store.similarity_search.call_count,1)

    def test_insufficient_evidence_returns_explicit_abstention_in_one_call(self):
        session,model,_=self.session('abstain')
        events=list(session.stream_answer('Unrelated question'))
        self.assertEqual(events[-1]['text'],rag.INSUFFICIENT_ANSWER)
        self.assertEqual(len(model.calls),1)

    def test_history_questions_help_retrieval_but_prior_answers_are_not_evidence(self):
        session,model,store=self.session()
        history=[{'role':'user','content':'Previous question'},{'role':'assistant','content':'untrusted-old-answer'}]
        list(session.stream_answer('And when?',history))
        self.assertIn('Previous question',store.similarity_search.call_args.args[0])
        payload=model.calls[0]['payload']
        self.assertEqual(payload['previous_questions'],['Previous question'])
        self.assertNotIn('untrusted-old-answer',json.dumps(payload))
        self.assertEqual(len(model.calls[0]['messages']),2)

    def test_duplicate_chunks_and_supports_do_not_duplicate_citation_links(self):
        document=Document(page_content='A subagent handles an isolated task.',metadata={'source':'https://example.com','chunk_id':1})
        session,_,_=self.session('duplicate_support',[document,document])
        result=list(session.stream_answer('Question'))[-1]
        self.assertEqual(len(result['evidence']),1)
        self.assertEqual(result['text'].count('](#'),1)

    def test_parallel_sessions_never_cite_another_users_anchor(self):
        def run(index):
            session,model,_=self.session()
            result=list(session.stream_answer('Question',turn_id=f'user-{index}'))[-1]
            self.assertEqual(len(model.calls),1)
            return result['text']
        with ThreadPoolExecutor(max_workers=5) as pool:answers=list(pool.map(run,range(5)))
        for i,answer in enumerate(answers):
            self.assertIn(f'#evidence-user-{i}-1',answer)
            for other in range(5):
                if other!=i:self.assertNotIn(f'#evidence-user-{other}-',answer)

    def test_tampered_quote_is_rejected_even_with_existing_id(self):
        answer=rag.CitedAnswer.model_validate({'claims':[{'text':'Claim','supports':[{'quote_id':'e1q1'}]}],'insufficient':False})
        evidence=[rag.Evidence(1,'https://example.com',1,'Actual source text.','anchor')]
        with self.assertRaises(rag.RagFlowError):
            rag._render_answer(answer,evidence,{'e1q1':rag.SourceSpan(1,'Invented quote')})

    def test_blank_question_never_retrieves_or_generates(self):
        session,model,store=self.session()
        with self.assertRaises(ValueError):list(session.stream_answer('  '))
        store.similarity_search.assert_not_called()
        self.assertEqual(model.calls,[])


class ConfigurationTests(unittest.TestCase):
    def test_credentials_are_redacted_and_blank_primary_key_falls_back(self):
        with patch.object(rag,'load_dotenv'),patch.dict(os.environ,{'GEMINI_API_KEY':'  ','GOOGLE_API_KEY':'secret-fallback'},clear=True):
            settings=rag.load_settings()
        self.assertEqual(settings.api_key,'secret-fallback')
        self.assertNotIn('secret-fallback',repr(settings))
        self.assertEqual(settings.chat_model,rag.DEFAULT_CHAT_MODEL)

    def test_missing_key_fails_before_preparation_network(self):
        with patch.object(rag,'build_index') as build:
            with self.assertRaises(rag.GeminiSetupError):rag.RagSession.prepare(rag.Settings())
        build.assert_not_called()

    def test_factory_disables_generation_retries_and_constructs_without_network(self):
        settings=rag.Settings(api_key='offline')
        with patch.object(rag,'ChatGoogleGenerativeAI') as factory:
            rag.RagSession(settings,MagicMock(),4)
        self.assertEqual(factory.call_args.kwargs['max_retries'],0)
        self.assertEqual(factory.call_args.kwargs['model'],settings.chat_model)
        self.assertFalse(factory.call_args.kwargs['vertexai'])
        self.assertEqual(factory.call_args.kwargs['timeout'],60)

    def test_embedding_preflight_needs_no_local_chat_model(self):
        settings=rag.local.Settings()
        with patch.object(rag.local,'Client') as client:
            api=client.return_value
            api.list.return_value.models=[SimpleNamespace(model=settings.embedding_model,digest='embedding-digest')]
            api.show.return_value.capabilities=['embedding']
            self.assertEqual(rag.local.check_ollama(settings,embeddings_only=True),'embedding-digest')
            api.show.assert_called_once_with(settings.embedding_model)

    def test_index_adapter_uses_existing_local_embedding_cache_contract(self):
        settings=rag.Settings(api_key='offline',embedding_gpu_layers=0)
        with patch.object(rag.local,'build_index',return_value=(MagicMock(),321)) as build:
            rag.build_index(settings,lambda _:None)
        self.assertTrue(build.call_args.kwargs['embeddings_only'])
        self.assertEqual(build.call_args.args[0].embedding_gpu_layers,0)
        self.assertEqual(build.call_args.args[0].embedding_model,settings.embedding_model)

    def test_safe_quota_errors_hide_provider_messages(self):
        error=ClientError(429,{'error':{'message':'secret-provider-detail','code':429}})
        self.assertIn('quota',rag.describe_error(error))
        self.assertNotIn('secret',rag.describe_error(error))

    def test_invalid_timeouts_and_request_limits_fail(self):
        for name,value in [('GEMINI_TIMEOUT_SECONDS','nan'),('OLLAMA_TIMEOUT_SECONDS','0'),
                           ('GEMINI_REQUESTS_PER_MINUTE','0'),('OLLAMA_EMBEDDING_GPU_LAYERS','-1')]:
            with self.subTest(name=name),patch.object(rag,'load_dotenv'),patch.dict(os.environ,{name:value},clear=True):
                with self.assertRaises(rag.GeminiSetupError):rag.load_settings()


class AdmissionTests(unittest.TestCase):
    def test_different_keys_in_same_quota_group_share_allowance(self):
        settings=rag.Settings(api_key='first',quota_group=uuid4().hex)
        self.assertIs(rag._get_limiter(settings),rag._get_limiter(replace(settings,api_key='second')))

    def test_rolling_limit_delays_excess_requests(self):
        limiter=rag.RequestLimiter(2,window_seconds=.04)
        def admitted(_):
            limiter.acquire()
            return monotonic()
        with ThreadPoolExecutor(max_workers=3) as pool:starts=sorted(pool.map(admitted,range(3)))
        self.assertGreaterEqual(starts[-1]-starts[0],.035)


if __name__=='__main__':unittest.main()
