# Changelog

## [0.1.0] - 2026-10-08

Initial working terminal RAG prototype.

### Added

- LangChain documentation loading, chunking, and source metadata.
- Gemini embeddings, vector retrieval, and evidence-analysis subagents.
- English answers with source links using Gemini 3.5 Flash-Lite by default.
- Project-root `.env` loading and validation of empty API keys.
- Resumable local vector indexes, bounded embedding batches, waits, and quota retries.
- UTF-8 terminal output and explicit chat API quota/overload failure messages.
- Setup instructions, tested dependency versions, and a safe environment template.

### Validation

- Live indexing, cache reuse, retrieval, and generation for the sample question passed on Windows with Python 3.13.
- Offline checks covered indexing/cache behavior and API failure handling.

### Current scope

- A fixed sample question and three documentation pages; no interactive chat UI or document uploads.
- Broader retrieval, citation, and answer-grounding evaluation remains future work.
