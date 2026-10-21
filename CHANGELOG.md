# Changelog

## [Unreleased]

### Added

- A retrieve-then-generate Gemini service with at most one generation attempt per question and local Ollama embeddings.
- Exact source-span IDs, native JSON-schema output and code-built citation links checked before publication.
- A reproducible five-session benchmark counting actual HTTP requests, token usage and per-question latency.
- Offline regressions for single-call generation, forged citations, abstention, safe errors, quota admission and session isolation.
- Python 3.13 CI on Ubuntu and Windows, without model servers or API credentials.

### Changed

- Make the single-request Gemini web workflow the portfolio's primary entry point; keep the earlier Ollama experiments available for comparison.
- Disable generation retries and SDK automatic function calling; return errors instead of silently adding repair requests.
- Share a process-local rolling request allowance across consistently configured sessions.
- Document installation from a fresh checkout, local embedding setup, architecture and measured limitations.
- Replace internal conversational notes and machine-specific workspace paths with technical project context.

### Experimental Ollama workflow

- Retain four isolated evidence-analysis tasks, mandatory file reads, exact-span extraction and validated claim citations.
- Retain CPU/iGPU placement options, local-only server helper and inference admission before timed requests.
- Preserve measured CPU-only and concurrent-user successes/failures in the validation reports.

### Validation

- The existing single-request validation completed five concurrent service questions with five Gemini HTTP requests and valid own-session citations.
- Two successful bursts had different latency ranges; results do not establish production throughput or a fixed response-time guarantee.
- See `docs/gemini-single-request-validation.md` for the dataset, request counts and limitations.

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
