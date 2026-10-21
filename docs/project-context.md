# Project context

## Purpose and scope

RAG Study Buddy is a personal portfolio prototype for documentation question answering with source evidence. The repository remains private until an explicit visibility change is requested. It is separate from coursework submissions.

The web app uses a fixed library of three public LangChain documentation pages. Authentication, uploads, persistent conversations and production deployment are outside the implemented scope.

## Active architecture

- `streamlit_app.py` uses `src/rag_study_buddy/gemini_rag.py`.
- Local Ollama `embeddinggemma:300m` creates document/query vectors. Saved indexes preserve source URLs, chunk IDs, content, embedding digest and endpoint identity.
- Code retrieves up to four distinct passages, assigns exact source-span IDs, and makes at most one native structured Gemini generation attempt.
- Default chat model: `gemini-3.5-flash-lite`; output cap 2048; timeout 60 seconds. SDK automatic function calling and retries are disabled.
- Code requires claim support, validates span membership and creates own-turn citation links. It withholds malformed answers and unchecked drafts; empty retrieval uses no generation request.
- Citation provenance is not proof of semantic correctness. Evaluate support and answer quality separately.
- Recent user questions help resolve follow-ups. Earlier assistant answers are not evidence.
- A process-local rolling allowance defaults to 12 requests/60 seconds. It does not coordinate external traffic, multiple processes, TPM or RPD.

## Interfaces and experiments

The active terminal sample is `python -m src.rag_study_buddy.gemini_agent`. The legacy `agent.py`/`rag.py` modules retain the four-subagent Ollama graph for regression tests and comparison. The `feat-ollama` branch preserves the local-model experiment.

Python 3.13 is the tested runtime. CI runs offline tests and dependency checks on Ubuntu and Windows; live published measurements were performed on Windows. Never require a real API key or model server for offline tests.

## Validation evidence

- Final saved-index single-request burst: five independent service sessions, five Gemini HTTP requests, five local query embeddings, zero retries/429/timeouts and 21 valid own-session citation links.
- Per-question latency: 2.03-2.95 seconds in the final burst and 8.93-15.47 seconds in an earlier successful burst. Neither is a production latency guarantee.
- Final chat usage: 9,038 input + 1,239 output = 10,277 tokens; provider and callback totals matched.
- The 321-chunk local index was reused without corpus re-embedding. Preparation retrieval probes were measured separately.
- Forty-six offline tests cover indexing, generation, citation rejection, UI behavior, isolation and legacy graph regressions.
- Detailed outcomes, including failures and limitations, remain in the validation reports. Raw traces, models and indexes stay in ignored local directories.

## Maintenance rules

Keep source metadata and evidence visible, do not publish unchecked drafts, and do not add silent model calls to repair output. Benchmark changes with real HTTP counts when request usage changes. Record only tests actually run. Use a public GitHub noreply identity for new commits and keep machine-specific paths and conversational logs out of shared documentation.

## Portfolio maintenance

The default main branch contains the single-request Gemini web workflow, fresh-checkout setup instructions, updated changelog and pinned-action Python 3.13 CI for Ubuntu/Windows. Current shared documentation omits machine-specific paths and internal conversational notes. Existing commit/tag history is intentionally retained; older author identities remain visible in that history. New local commits use the GitHub noreply identity. Repository visibility remains private.
