# Changelog

## [0.1.3] - 2026-10-09

### Added

- Streamlit chat UI with a selectable documentation library and evidence panel.
- Numbered citations with passage popovers, source links, and per-answer evidence selection.
- Coordinator answer streaming, recent conversation context, and processing/error states.
- Offline regression tests for service behavior and Streamlit interactions.

### Changed

- Extracted the terminal prototype into a reusable RAG service with import-safe entry points.
- Retained the existing index identity, bounded batching/retries, and resumable cache behavior.
- Isolated evidence files per answer and prepared services per browser session.
- Replaced conversation avatars with **Question:** and **Answer:** text labels, including streamed responses.

### Validation

- All 13 offline regression tests passed, including conversation labels, citation controls, evidence selection, retrieval metadata, and generation failure handling.
- No new live Gemini calls were made for this release.

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
