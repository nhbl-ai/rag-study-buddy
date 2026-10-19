# RAG Study Buddy

The `feat-ollama` branch runs the 0.1.3 application with local Ollama chat and embeddings. Code enforces retrieval, isolated evidence subagent execution, and citation validation before publishing an English answer.

## What it does

- Selects three LangChain documentation pages, splits text, and preserves source URLs and chunk IDs.
- Uses `embeddinggemma:300m` with `InMemoryVectorStore` and a resumable disk index.
- Uses `qwen3.5:4b` for both coordinator and evidence-analysis subagent.
- Shows Question:/Answer: labels, progress updates, citation popovers and the retrieved evidence for each turn. Unchecked answer drafts and private analyst output are never displayed.
- Keeps up to six recent successful turns; conversation and evidence are isolated per browser session.
- Requires no Gemini key. Existing Gemini variables in `.env` are ignored. LangSmith tracing is disabled.

## Setup on Windows

Run from the repository root in PowerShell (Python 3.13):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Install [Ollama for Windows](https://ollama.com/download/windows). During migration it is installed in the ignored `work/ollama-runtime/` directory. The helper also supports a standard Ollama installation.

Start the local server in a separate terminal and leave it running:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ollama.ps1
```

The helper binds to `127.0.0.1:11434`, disables Ollama cloud features, sets one parallel inference slot, and stores models in ignored `work/ollama-models/`. If the Ollama desktop app already occupies that port, quit it first. These settings apply only to the helper's process; global environment variables are not modified.

In another terminal, download the two models. When installed by this migration, use:

```powershell
.\work\ollama-runtime\ollama.exe pull qwen3.5:4b
.\work\ollama-runtime\ollama.exe pull embeddinggemma:300m
```

For a standard installation, use `ollama pull ...` instead. Downloads are about 4 GB and 622 MB, plus the runtime. The helper's model directory stays inside this repository and is ignored by Git.

Defaults work even if `.env` contains only the old Gemini settings. Optional overrides in the project-root `.env`:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen3.5:4b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:300m
OLLAMA_NUM_CTX=8192
OLLAMA_TIMEOUT_SECONDS=180
```

The chat model must support text generation and Ollama JSON-schema output; the embedding model must support embeddings. Tool selection is controlled by application code. Chat uses `reasoning=False` and an output limit of 1024 tokens per call; sampling otherwise uses model defaults. Preparation checks downloaded models and their capabilities before downloading documents. Model downloads are explicit; the application does not pull models automatically. Cloud model tags are rejected.

8192 tokens is a starting context size for the existing agent flow, not a promise that all conversations fit. Longer context increases memory use. Close heavy applications on a 16 GB RAM machine; use `ollama ps` to inspect GPU offloading. No fixed concurrency capacity is claimed for the laptop.

## Run the chat UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Open `http://127.0.0.1:8501`. Choose documents and click **Prepare documents**, or ask a question. No API key is required. The initial render makes no document or model requests.

The left column shows the conversation; **Evidence** shows actual passages with chunk IDs and source links. Numbered controls open citation popovers. **View evidence** selects an answer's passages; inline citation links jump to them. Retrieved passages still need checking against the answer's claims.

**New conversation** retains the prepared index. Changing documents or model settings resets the prepared service and conversation. A new browser session prepares again and can reuse the disk index. Restart Streamlit after editing `.env` to reload environment settings.

## Run the terminal sample

```powershell
.\.venv\Scripts\python.exe .\src\rag_study_buddy\agent.py
```

It prints indexing progress, retrieved passages, and an answer to the existing sample question. Terminal citation links use source URLs. Both entry points use the same RAG service.

Indexing saves each batch of at most 80 passages, with no Gemini quota pauses. A failed batch leaves previous batches available for resuming. Cache identity includes provider, endpoint, embedding model and its digest, content, and source metadata. Ollama never reuses Gemini vectors. Changing only the chat model does not require re-embedding. Existing Gemini indexes remain untouched.

The coordinator now follows fixed steps instead of letting the model decide whether to call tools. `search_documents` retrieves up to four passages, retaining URLs and chunk IDs. Every distinct passage is assigned sequentially through the real Deep Agents `task` tool to an isolated `evidence-analyst` LangGraph subgraph. The subgraph reads its assigned in-memory file before asking the model to select numbered source spans. Code extracts the selected spans verbatim; the model does not retype quotations. Only selected spans reach synthesis.

Synthesis returns structured claims referencing one unique `quote_id` per supporting quote. Its JSON schema enumerates only IDs actually supplied to synthesis. The application rejects unknown evidence, unverified quotes, missing support and model-written citation links, then renders canonical numbered links itself. These checks verify provenance and citation structure; semantic support still needs evaluation. Empty or irrelevant evidence produces an explicit insufficient-information answer. A failed subagent stops synthesis. Answers appear after validation rather than token by token; progress and evidence events remain available.

Chat and embeddings are local, but preparation still downloads the three public documentation pages, including when reusing an index. This version is not completely offline. Authentication, uploads, persistent chats, and deployment for concurrent users have not been added.

## Workflow

```text
Load settings -> Check Ollama models -> Download documents -> Split text
              -> Load or build Ollama vector index
              -> Required search_documents retrieval (top 4)
              -> Required task -> isolated evidence-analyst -> read_file -> exact quotes
              -> Structured answer claims -> validate quotes and evidence IDs
              -> Render canonical citations -> publish complete answer
              -> UI shows the retrieved passages for this turn
```

## Validation

Run offline regression checks without a model server or API key:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests separate indexing/retrieval metadata and cache behavior from graph/generation and UI behavior. They exercise the real Deep Agents task tool and mandatory file-reading subgraph using a fake model, plus history, citation validation, safe failures, and UI configuration changes. Passing them does not establish the local model's answer quality. Historical skipped-step failures and the subsequent CPU-only rerun are recorded separately in `docs/ollama-validation.md`.

The historical 2026-10-08 Gemini terminal validation and 0.1.3 synthetic browser checks remain documented in project context; they are not evidence of live Ollama quality. Broader retrieval, citation grounding and concurrent load evaluation remain necessary.

## Structure

```text
streamlit_app.py             Chat and evidence UI
scripts/start_ollama.ps1     Local-only server helper
src/rag_study_buddy/rag.py   Indexing, retrieval and generation
src/rag_study_buddy/agent.py Terminal sample
requirements.txt            Python dependencies
.env.example                Shareable local-model configuration
CHANGELOG.md                Release notes
docs/project-context.md     Decisions and validation
tests/                      Offline regressions
data/processed/             Generated indexes; ignored
work/                       Local runtime, models and test output; ignored
```

Credentials, `.env`, private documents, model files, and indexes must stay out of Git.
