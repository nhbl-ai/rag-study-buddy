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

## Optional CPU + Intel iGPU profile

The measured profile on this laptop uses eight chat layers on Intel and the remaining 25 on CPU, CPU embeddings, one actual inference slot, and a 512 MiB prompt cache. Layer counts are model-specific, not CPU/GPU utilization percentages. Automatic placement remains the default when layer overrides are absent.

Start the separate profile in one terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ollama.ps1 -IntelHybrid -ParallelSlots 1 -Port 11435 -PromptCacheMiB 512
```

In another terminal, point the application at that server and set placement on its model requests:

```powershell
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11435'
$env:OLLAMA_CHAT_GPU_LAYERS = '8'
$env:OLLAMA_EMBEDDING_GPU_LAYERS = '0'
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py --server.port 8502
```

The helper selects Vulkan device 0 by default (`-VulkanDevice` changes the index), enables the iGPU, excludes CUDA/ROCm/HIP devices and limits loaded models to two. Verify Intel in the server log; device numbering can differ on other machines. The positive layer count uses the server's selected GPU; it does not itself select Intel. Blank layer settings use automatic placement; zero requests CPU only. These are process-local settings, and the existing `.env` is not rewritten.

`-ParallelSlots` requests one to five slots, but Ollama 0.40.5 forces this `qwen35` model to **one actual slot**, including when two were requested. The application also admits one chat request at a time per endpoint/model across sessions in the same Python process, so queued users wait before the timed HTTP request begins. Both analyst and synthesis calls share that slot, and exceptions release it. Queue time is included in end-to-end latency; the unchanged 180s timeout does not bound a user's total waiting time. This admission control is process-local, not distributed. Five clients can submit work concurrently and queue; raising this option does not bypass that runtime restriction. The 512 MiB cache bound uses the bundled llama-server's `LLAMA_ARG_CACHE_RAM`; verify its effective limit in the log if changing runtime versions. Context remains 8192 and request timeout remains 180 seconds. See [the tuning measurements](docs/ollama-hybrid-tuning.md) for outcomes and limitations.

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

The [initial five-user CPU+iGPU burst](docs/ollama-concurrency-validation.md) returned four accepted responses and one timeout. The [tuned profile with application admission](docs/ollama-hybrid-tuning.md) completed all five workflows without timeout in 641.89s, with 20 subagents/reads and 19 valid citation links; individual waits were about 6.1–10.7 minutes. One comparison answer still needs semantic revision despite valid citations. All 28 offline tests passed. These measurements do not establish responsive multi-user capacity.

A [dedicated single-user CPU+iGPU run](docs/ollama-hybrid-single-user.md) completed the primary question in **119.14s**, with four subagents/reads and three valid citation links. Saved-index/model preparation added **12.32s**; the measured total was **131.46s**, excluding downloads and fresh indexing. This is one run with the selected eight-layer Intel profile.

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
