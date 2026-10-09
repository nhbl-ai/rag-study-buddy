# RAG Study Buddy

Version **0.1.3** of RAG Study Buddy answers questions using LangChain documentation, Deep Agents, and the Gemini Developer API. It includes a Streamlit chat interface and a terminal sample.

## What it does

- Lets you choose from three documentation pages and splits them into chunks.
- Preserves source URLs and chunk IDs.
- Uses `gemini-embedding-001` and an in-memory vector store for retrieval.
- Saves a resumable index in `data/processed/` so unchanged documents are not embedded again.
- Uses an evidence-analysis subagent and `gemini-3.5-flash-lite` to generate an English answer.
- Shows a conversation with **Question:** and **Answer:** labels next to the passages actually retrieved for each answer.
- Streams the coordinator's answer, with numbered citations, passage popovers, and source links.
- Keeps recent conversation context, and lets you inspect earlier answers' evidence.
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

## Run the chat UI

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Open `http://127.0.0.1:8501` in your browser. Choose documents in the sidebar, then click **Prepare documents** or ask your first question. If `.env` has no API key, enter one in the sidebar's password field. The UI never writes that key to a file.

The conversation appears on the left; **Evidence** on the right shows retrieved passages, chunk IDs, and source links. Click a numbered citation control below an answer to read its passage in a popover, or **View evidence** to select that answer in the panel. Inline citation links jump to passages when that answer is selected. Retrieved passages are reference material; they do not by themselves prove that every answer claim is correct.

**New conversation** clears chat and evidence while retaining the prepared index in the current browser session. Changing selected documents or credentials resets the conversation and prepared session; saved indexes remain available. Reloading into a new browser session requires preparing documents again, reusing the saved index when it matches.

The UI shows download, indexing, retrieval, generation, and error states. It initializes the RAG service only when you prepare documents or submit a question. The initial page does not download documents or call Gemini. Answers use up to six recent successful turns as conversation context; each new question retrieves fresh evidence.

The server binds to localhost. This version selects the bundled documentation pages; document uploads, authentication, and persistent chat storage are outside its current scope.

## Run the terminal sample

```powershell
.\.venv\Scripts\python.exe .\src\rag_study_buddy\agent.py
```

The script prints download/indexing progress, the passages retrieved by the agent with their sources, and the final answer. Change the sample question in `src/rag_study_buddy/agent.py` to try another question. Both entry points use the same service in `src/rag_study_buddy/rag.py`; default documentation pages are defined in `SOURCE_PAGES` there. The terminal converts numbered citation jump links into source URLs.

The first run indexes documents in batches of at most 80 chunks with 65-second pauses. Indexing can take several minutes. Each successful batch is saved; rerunning resumes an incomplete index. Later runs reuse the completed index when document content, source metadata, and the embedding model match. They still download documents and call Gemini for query embeddings and generation.

The batch settings address the 100 requests/minute embedding limit observed during validation. Gemini quotas vary by project. The script retries embedding quota failures with a bounded wait, and reports chat quota or service-overload failures. You can select another available chat model through `GEMINI_CHAT_MODEL` without rebuilding the document index.

## Workflow

```text
Load .env -> Download documents -> Split into chunks
          -> Load or build the cached vector index
          -> Main agent searches for evidence
          -> Evidence subagent reads retrieved files
          -> Main agent streams an answer with numbered citations
          -> UI shows the retrieved passages for this turn
```

## Validation

On 2026-10-08, a live run indexed 322 chunks, reused the saved index, retrieved relevant subagent documentation, and answered the sample question with a source link (exit code 0). Retrieval was checked separately from generation. Offline checks covered batching, retries, cache reuse, resuming after failure, cache invalidation, and API failure messages.

Version 0.1.3 includes offline regression checks for cache reuse/resuming, retrieval metadata, real graph/subagent file access with a fake model, stream filtering, chat history, evidence selection, document changes, and failure/retry states. Run them without a Gemini key:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests use mocked downloads, embeddings, or models, and do not call Gemini. Retrieval/index behavior is checked separately from answer generation and UI behavior. The existing live result above belongs to the terminal v0.1.0 baseline; it is not a live validation of the new UI. Citation completeness and answer grounding still need broader evaluation.

The UI was also run locally and checked in headless Chrome at desktop and mobile viewport sizes, including citation popovers. Chat content for these visual checks was synthetic; no Gemini calls were made.

## Structure

```text
streamlit_app.py             Streamlit chat and evidence interface
.streamlit/config.toml       Local server and visual theme settings
src/rag_study_buddy/rag.py    Shared indexing, retrieval, and generation service
src/rag_study_buddy/agent.py  Terminal sample entry point
src/rag_study_buddy/test.py    Standalone document-download check
requirements.txt             Versions used by the validated environment
.env.example                 Credential/configuration template
CHANGELOG.md                 Release notes
docs/project-context.md      Project context and decisions
AGENTS.md                    Collaboration guidelines
data/processed/              Generated local index; ignored by Git
tests/                       Offline service and UI regression checks
```

`.env`, `.venv`, private documents, and generated indexes are excluded from Git. Only source code, documentation, and shareable configuration templates belong in the repository.
