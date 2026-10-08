# RAG Study Buddy

Version **0.1.0** is a working terminal RAG prototype built with Deep Agents and the Gemini Developer API. It answers a fixed sample question using LangChain documentation and includes a source link.

## What it does

- Downloads three documentation pages and splits them into chunks.
- Preserves source URLs and chunk IDs.
- Uses `gemini-embedding-001` and an in-memory vector store for retrieval.
- Saves a resumable index in `data/processed/` so unchanged documents are not embedded again.
- Uses an evidence-analysis subagent and `gemini-3.5-flash-lite` to generate an English answer.
- Loads credentials from the project-root `.env` or a hidden prompt; LangSmith tracing is disabled.

## Setup

The prototype was validated on Windows with Python 3.13. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set your Gemini API key:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
```

`GOOGLE_API_KEY` is also accepted when `GEMINI_API_KEY` is not set. If neither variable contains a key, the script prompts for one. Never commit real credentials.

## Run

```powershell
.\.venv\Scripts\python.exe .\src\rag_study_buddy\agent.py
```

The script prints download/indexing progress, four retrieved passages with sources, and the final answer. Change `question` and `source_urls` in `src/rag_study_buddy/agent.py` to try another question or documentation page.

The first run indexes documents in batches of at most 80 chunks with 65-second pauses. Indexing can take several minutes. Each successful batch is saved; rerunning resumes an incomplete index. Later runs reuse the completed index when document content, source metadata, and the embedding model match. They still download documents and call Gemini for query embeddings and generation.

The batch settings address the 100 requests/minute embedding limit observed during validation. Gemini quotas vary by project. The script retries embedding quota failures with a bounded wait, and reports chat quota or service-overload failures. You can select another available chat model through `GEMINI_CHAT_MODEL` without rebuilding the document index.

## Workflow

```text
Load .env -> Download documents -> Split into chunks
          -> Load or build the cached vector index
          -> Print a retrieval check
          -> Main agent searches for evidence
          -> Evidence subagent reads retrieved files
          -> Main agent answers with a source link
```

## Validation

On 2026-10-08, a live run indexed 322 chunks, reused the saved index, retrieved relevant subagent documentation, and answered the sample question with a source link (exit code 0). Retrieval was checked separately from generation. Offline checks covered batching, retries, cache reuse, resuming after failure, cache invalidation, and API failure messages.

This is an initial prototype, not a comprehensive RAG quality evaluation. There is no interactive chat loop, document-upload flow, or graphical interface yet. Citation completeness and answer grounding still need broader evaluation.

## Structure

```text
src/rag_study_buddy/agent.py   Main single-file prototype
src/rag_study_buddy/test.py    Standalone document-download check
requirements.txt             Versions used by the validated environment
.env.example                 Credential/configuration template
CHANGELOG.md                 Release notes
docs/project-context.md      Project context and decisions
AGENTS.md                    Collaboration guidelines
data/processed/              Generated local index; ignored by Git
```

`.env`, `.venv`, private documents, and generated indexes are excluded from Git. Only source code, documentation, and shareable configuration templates belong in the repository.
