"""Measure simultaneous single-request Gemini RAG sessions on a saved public index."""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from src.rag_study_buddy import gemini_rag as rag

QUESTIONS = [
    "What is a subagent, and when should I use one?",
    "How do subagents prevent context bloat in a deep agent?",
    "When should I avoid using subagents?",
    "How does a CompiledSubAgent differ from a dictionary-based SubAgent?",
    "How can I stream a deep agent's response while distinguishing subagent activity?",
]


def safe_error(error):
    cause, seen = error, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        code = getattr(cause, 'code', None)
        if code is not None:
            return {'type': type(error).__name__, 'cause_type': type(cause).__name__, 'code': code}
        cause = cause.__cause__ or cause.__context__
    return {'type': type(error).__name__}


def usage_totals(calls):
    values = [c['usage'] for c in calls if c.get('usage')]
    fields = ('prompt_token_count','tool_use_prompt_token_count','candidates_token_count',
              'thoughts_token_count','cached_content_token_count','total_token_count')
    result = {field: sum(v.get(field,0) or 0 for v in values) for field in fields}
    result['input_tokens'] = result['prompt_token_count'] + result['tool_use_prompt_token_count']
    result['output_tokens'] = result['candidates_token_count'] + result['thoughts_token_count']
    result['responses_with_usage'] = len(values)
    result['calls_without_usage'] = len(calls)-len(values)
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('work/gemini-single-five'))
    parser.add_argument('--users',type=int,choices=range(1,6),default=5)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error('Output directory already exists; choose a fresh --output to preserve earlier results.')
    out.mkdir(parents=True)
    settings = replace(rag.load_settings(),embedding_gpu_layers=0)
    index = args.index.resolve()
    lock = threading.RLock()
    report = {'users': [],'retrieval_probes': [],'preparation_embedding_http': [],
              'model': settings.chat_model,'embedding_model': settings.embedding_model,
              'embedding_placement': 'CPU requested with num_gpu=0',
              'scope': 'independent prepared RagSession objects; five different questions; no browser/network-user load simulation',
              'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(rag.__file__).read_bytes()).hexdigest(),
              'native_chat_max_retries': 0,'process_request_allowance_per_minute': settings.requests_per_minute}
    sessions, embeddings_clients = [], []

    def save(name='partial.json'):
        with lock:
            text = json.dumps(report,ensure_ascii=False,indent=2,default=str)
        temp = out/(name+'.tmp')
        temp.write_text(text,encoding='utf-8')
        temp.replace(out/name)

    def instrument_http(http,rows):
        original = http.send
        def send(request,*a,**kw):
            item={'path': request.url.path,'method': request.method,'started': time.perf_counter(),'status': 'running'}
            with lock: rows.append(item)
            try:
                response=original(request,*a,**kw)
                with lock:item.update(status='response',status_code=response.status_code)
                return response
            except Exception as error:
                with lock:item.update(status='transport_error',error=safe_error(error))
                raise
            finally:
                with lock:item['seconds_to_headers']=time.perf_counter()-item['started']
        http.send=send

    def instrument_chat(model,row):
        sdk=model.client.models
        stream_original=sdk.generate_content_stream
        original=sdk.generate_content
        def begin():
            item={'started':time.perf_counter(),'status':'running'}
            with lock:row['sdk_calls'].append(item)
            return item
        def capture(item,response):
            if response.usage_metadata is not None:
                with lock:item['usage']=response.usage_metadata.model_dump(exclude_none=True)
        def generate(*a,**kw):
            item=begin()
            try:
                response=original(*a,**kw);capture(item,response)
                with lock:item['status']='success'
                return response
            except Exception as error:
                with lock:item.update(status='error',error=safe_error(error))
                raise
            finally:
                with lock:item['seconds']=time.perf_counter()-item['started']
        def stream(*a,**kw):
            item=begin()
            try:
                for response in stream_original(*a,**kw):
                    capture(item,response)
                    yield response
                with lock:item['status']='success'
            except Exception as error:
                with lock:item.update(status='error',error=safe_error(error))
                raise
            finally:
                with lock:item['seconds']=time.perf_counter()-item['started']
        sdk.generate_content=generate
        sdk.generate_content_stream=stream
        instrument_http(model.client._api_client._httpx_client,row['generation_http'])

    class Recorder(BaseCallbackHandler):
        def __init__(self,row):self.row=row
        def on_chat_model_start(self,serialized,messages,**kw):
            with lock:
                self.row['model_starts']+=1
                self.row['admission_wait_seconds']=kw.get('metadata',{}).get('request_admission_wait_seconds')
        def on_llm_end(self,response,**kw):
            message=response.generations[0][0].message
            with lock:
                self.row['callback_usage']=message.usage_metadata
                self.row['model_output']=message.content
        def on_tool_start(self,*a,**kw):
            with lock:self.row['tool_calls']+=1

    def embedding_client(rows):
        embedding=OllamaEmbeddings(model=settings.embedding_model,base_url=settings.base_url,num_gpu=0,
                                   client_kwargs={'timeout':settings.embedding_timeout_seconds})
        embeddings_clients.append(embedding)
        instrument_http(embedding._client._client,rows)
        return embedding

    def worker(row,session,barrier):
        recorder=Recorder(row)
        barrier.wait()
        with lock:row.update(started=time.perf_counter(),status='running')
        try:
            for event in session.stream_answer(row['question'],(),turn_id=f"single-{out.name}-user-{row['user']}",config={'callbacks':[recorder]}):
                with lock:
                    if event['type']=='status':row['stage']=event['text']
                    elif event['type']=='evidence':row['evidence']=[asdict(e) for e in event['evidence']]
                    elif event['type']=='draft':row['draft_published']=True
                    elif event['type']=='complete':row.update(completed=True,answer=event['text'],evidence=[asdict(e) for e in event['evidence']])
        except Exception as error:
            with lock:row['error']=safe_error(error)
        finally:
            with lock:
                row['seconds']=time.perf_counter()-row['started']
                row['usage']=usage_totals(row['sdk_calls'])
                links=re.findall(r'\[(\d+)\]\(#([^)]+)\)',row.get('answer',''))
                actual={(str(e['number']),e['anchor']) for e in row.get('evidence',[])}
                callback=row.get('callback_usage') or {}
                row['usage_crosscheck']=all(callback.get(field,0)==row['usage'][other] for field,other in
                    [('input_tokens','input_tokens'),('output_tokens','output_tokens'),('total_tokens','total_token_count')])
                row['request_counts']={'generation_sdk_calls':len(row['sdk_calls']),'generation_http_attempts':len(row['generation_http']),
                    'generation_http_statuses':dict(Counter(str(h.get('status_code',h['status'])) for h in row['generation_http'])),
                    'local_embedding_http_attempts':len(row['embedding_http'])}
                row['checks']={'completed':row.get('completed',False),'one_generation_request':len(row['generation_http'])==len(row['sdk_calls'])==row['model_starts']==1,
                    'generation_http_success':all(h.get('status_code')==200 for h in row['generation_http']),
                    'one_local_embedding':len(row['embedding_http'])==1 and row['embedding_http'][0].get('status_code')==200,
                    'valid_own_citation_links':bool(links) and all(link in actual for link in links),
                    'no_unchecked_draft':not row.get('draft_published',False),'no_agent_tool_loop':row['tool_calls']==0}
                row['citation_count']=len(links)
                row['workflow_passed']=all(row['checks'].values())
                row['fully_answered']=row.get('completed',False) and rag.INSUFFICIENT_ANSWER not in row.get('answer','')
                row['status']='completed' if row.get('completed') else 'failed'
                if row.get('answer'):(out/f"answer-user-{row['user']}.md").write_text(row['answer'],encoding='utf-8')
                print('USER DONE',json.dumps({k:row.get(k) for k in ('user','seconds','workflow_passed','fully_answered','request_counts','citation_count','usage','error')}),flush=True)

    try:
        if not settings.api_key:raise rag.GeminiSetupError('Gemini credential is not configured.')
        digest=rag.local.check_ollama(rag.local.Settings(base_url=settings.base_url,embedding_model=settings.embedding_model),embeddings_only=True)
        prep=embedding_client(report['preparation_embedding_http'])
        base_store=InMemoryVectorStore.load(str(index),embedding=prep)
        assert {r['metadata']['source'] for r in base_store.store.values()}<=set(rag.SOURCE_PAGES.values()),'Benchmark accepts only the bundled public documentation corpus'
        identity={'provider':'ollama','embedding_model':settings.embedding_model,'embedding_digest':digest,'base_url':settings.base_url,
                  'chunks':[{'text':r['text'],'metadata':r['metadata']} for r in base_store.store.values()]}
        expected=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()[:20]
        assert index.stem=='rag-index-'+expected,'Embedding index identity does not match the configured model/digest/endpoint'
        report.update(chunks=len(base_store.store),embedding_digest=digest,index_identity_verified=True)
        for question in QUESTIONS[:args.users]:
            began=time.perf_counter();matches=base_store.similarity_search_with_score(question,k=4)
            report['retrieval_probes'].append({'question':question,'seconds':time.perf_counter()-began,
                'matches':[{'source':d.metadata['source'],'chunk_id':d.metadata['chunk_id'],'score':float(score),'text':d.page_content} for d,score in matches]})
        print('SEPARATE RETRIEVAL',[(round(p['seconds'],3),[m['chunk_id'] for m in p['matches']]) for p in report['retrieval_probes']],flush=True)
        for i,question in enumerate(QUESTIONS[:args.users],1):
            row={'user':i,'question':question,'status':'ready','sdk_calls':[],'generation_http':[],'embedding_http':[],'model_starts':0,'tool_calls':0}
            report['users'].append(row)
            embedding=embedding_client(row['embedding_http'])
            store=InMemoryVectorStore.load(str(index),embedding=embedding)
            session=rag.RagSession(settings,store,len(store.store));sessions.append(session)
            instrument_chat(session.model,row)
        barrier=threading.Barrier(args.users+1)
        with ThreadPoolExecutor(max_workers=args.users) as pool:
            futures=[pool.submit(worker,row,session,barrier) for row,session in zip(report['users'],sessions)]
            epoch=time.perf_counter();barrier.wait();pending=set(futures)
            print('DISPATCH',args.users,'users simultaneously',flush=True)
            while pending:
                done,pending=wait(pending,timeout=15)
                for future in done:future.result()
                print('PROGRESS',round(time.perf_counter()-epoch,1),[(u['user'],u['status']) for u in report['users']],flush=True)
                save()
        report['batch_seconds']=max(u['started']+u['seconds'] for u in report['users'])-epoch
        report['dispatch_skew_seconds']=max(u['started'] for u in report['users'])-min(u['started'] for u in report['users'])
    except Exception as error:
        report['harness_error']=safe_error(error)
        print('HARNESS ERROR',report['harness_error'],flush=True)
    finally:
        calls=[c for u in report['users'] for c in u['sdk_calls']]
        report['chat_usage_total']=usage_totals(calls)
        report['summary']={'completed':sum(u.get('completed',False) for u in report['users']),
            'workflow_passed':sum(u.get('workflow_passed',False) for u in report['users']),
            'fully_answered':sum(u.get('fully_answered',False) for u in report['users']),
            'generation_http_attempts':sum(len(u['generation_http']) for u in report['users']),
            'generation_http_statuses':dict(Counter(str(h.get('status_code',h['status'])) for u in report['users'] for h in u['generation_http'])),
            'local_question_embedding_http':sum(len(u['embedding_http']) for u in report['users']),
            'citation_links':sum(u.get('citation_count',0) for u in report['users'])}
        save('results.json')
        for session in sessions:
            try:session.model.client.close()
            except Exception:pass
        for embedding in embeddings_clients:
            try:embedding._client._client.close()
            except Exception:pass
        print('SUMMARY',json.dumps({k:report.get(k) for k in ('batch_seconds','summary','chat_usage_total','harness_error')}),flush=True)
    return 0 if report['summary']['workflow_passed']==args.users else 1


if __name__=='__main__':sys.exit(main())
