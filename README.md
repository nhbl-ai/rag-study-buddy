# RAG Study Buddy

The `feat-gemini-single-request` branch uses local Ollama embeddings and at most **one Gemini generation attempt per question**. Retrieval runs in code before generation. Gemini returns structured claims referencing source-span IDs; code validates IDs and builds citation links before publishing the answer.

The separate `feat-ollama` branch retains the measured four-subagent local workflow. The Gemini application does not execute analyst subagents or tool-call loops.

## Setup

Run from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the project-root `.env`. The key stays on the application server and is not requested in the browser. The initial UI render makes no model/document requests; preparing or asking requires a configured key. Never commit `.env` or credentials.

Start the existing local Ollama server, using the installed runtime/helper when needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ollama.ps1
```

Only `embeddinggemma:300m` is required locally. With the repository-local runtime, download it if missing:

```powershell
.\work\ollama-runtime\ollama.exe pull embeddinggemma:300m
```

For a standard Ollama installation use `ollama pull embeddinggemma:300m`. Models/indexes remain ignored. Existing downloaded models and valid Ollama indexes can be reused; no local Qwen chat model is needed by this branch's Gemini UI. Model download is explicit.

Configuration examples are in [.env.example](.env.example). Defaults are Gemini `gemini-3.5-flash-lite`, chat timeout 60s, local embedding timeout 180s, and a process-local allowance of 12 generation requests per 60 seconds. `OLLAMA_EMBEDDING_GPU_LAYERS=0` requests CPU embeddings; blank uses automatic placement. Old `OLLAMA_CHAT_MODEL`, chat GPU-layer and context overrides do not configure Gemini generation. No `.env` is rewritten by the application.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Open `http://127.0.0.1:8501`, select documents and prepare or ask a question. The web interface preserves Question:/Answer: labels, evidence panels/popovers, source links, per-turn anchors and conversation resets. Answers appear only after validation, rather than as unchecked token drafts.

The terminal sample for this branch is:

```powershell
.\.venv\Scripts\python.exe -m src.rag_study_buddy.gemini_agent
```

The older `agent.py` and `rag.py` retain the Ollama graph for comparison/regressions. They are not the active Gemini web entry point.

## Workflow and citation checks

1. Preparation checks the local embedding model, downloads the selected public documentation pages, splits them, and loads/builds the saved Ollama index. Identity includes embedding digest, endpoint, content and metadata. Gemini vectors are never mixed with local vectors.
2. Each question performs one local embedding/retrieval, deduplicating up to four passages and preserving source URL/chunk ID. Up to two prior user questions can help resolve follow-ups; old assistant answers are not supplied as evidence.
3. Code splits the retrieved text into exact source spans with unique IDs. The JSON schema permits only those IDs. No analyst model call is made.
4. A shared request allowance admits generation. The application makes one native JSON-schema Gemini call, with automatic function calling and HTTP retries disabled. It does not automatically regenerate malformed output or repair citations.
5. Code rejects invalid JSON/schema, unsupported IDs, missing supports, altered source spans and model-written citation links/URLs/HTML in claim text. It constructs canonical links from actual evidence anchors and publishes the validated answer. Empty retrieval uses zero Gemini requests; irrelevant evidence may yield an explicit insufficient-information answer from the single request.

These checks prove source membership, exact-span provenance and citation structure, **not semantic entailment**. Claims still require quality evaluation against the source. The UI exposes actual passages for that review. A failed request/validation yields an error and retains evidence; a user-initiated retry is a new attempt and consumes another request.

The allowance is a rolling request budget, not an inference-slot limit: five initial requests can overlap. It is shared across browser sessions and keys with the same model/quota-group configuration in one Python process. It does not coordinate separate server processes or account for other applications, TPM or RPD. Keep `GEMINI_QUOTA_GROUP` common for keys belonging to the same project and configure limits consistently. Queue waits count toward user latency and occur before the timed generation request. Provider 429/503 errors are shown safely without automatic retry.

Chat generation is online. Query/corpus embedding and vector search are local; preparation also downloads public documents. No authentication, uploads, persistent chats or production deployment is introduced in this experiment. Client machines only need a browser once the server is hosted.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Offline checks cover one-call generation, invalid citations, withholding unchecked answers, abstention, source isolation between concurrent sessions, credential-safe errors, shared admission and the existing indexing/UI/legacy Ollama behavior.

A real five-user API benchmark can be reproduced with a matching saved local embedding index:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_gemini_single.py --index data/processed/rag-index-bebd0f5eda4a89d7bd62.json --output work/gemini-single-five-new --users 5
```

The filename above is the index on the tested machine; use the matching index created by preparation on another machine. The benchmark checks model digest/index identity and accepts only the bundled public-documentation corpus. Choose a fresh output folder for every run. It makes real Gemini requests using the configured key. It checks retrieval separately and counts actual generation HTTP attempts, local embedding requests, SDK usage, callbacks, citations and latency per question. Raw results stay under ignored `work/`.

See [the single-request results](docs/gemini-single-request-validation.md) and [the earlier agentic API comparison](docs/gemini-api-one-two-users.md). Five completed questions in one burst do not establish a maximum user count or a production latency guarantee.

Credentials, private documents, model files, vector indexes and large benchmark logs must stay out of Git.
