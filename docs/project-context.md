# Project context

## Purpose and scope

RAG Study Buddy is a personal portfolio prototype for documentation question answering with source evidence. The repository remains private until an explicit visibility change is requested. It is separate from coursework submissions.

The web app uses a fixed library of three public LangChain documentation pages. Authentication, uploads, persistent conversations and production deployment are outside the implemented scope.

The `introse/smart-study-planner-rag` branch packages the Python RAG component for use alongside the Smart Study Planner backend. Python code remains under `src/` and `tests/`. Planner HTTP integration and access-controlled document processing are separate work.

## Active architecture

- `streamlit_app.py` uses `src/rag_study_buddy/gemini_rag.py`.
- Local Ollama `embeddinggemma:300m` creates document/query vectors. Saved indexes preserve source URLs, chunk IDs, content, embedding digest and endpoint identity.
- Code retrieves passages once using `RAG_RETRIEVAL_TOP_K` (four by default), deduplicates them and assigns exact source-span IDs. Generation tries configured providers in `RAG_PROVIDER_ORDER` (Gemini, Groq, Cloudflare Workers AI by default), with a single attempt per provider.
- Gemini uses native structured output with a span-ID enum. Groq Chat Completions uses strict structured output and Cloudflare Workers AI uses JSON Mode over the direct REST API with a bearer token and Account ID. All three use the same local answer and span validation.
- Default models are `gemini-3.5-flash-lite`, `openai/gpt-oss-20b`, and `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. `RAG_MAX_OUTPUT_TOKENS` defaults to 2048 tokens and the default timeout is 60 seconds per provider. Gemini SDK automatic function calling and retries are disabled.
- `rag.toml` holds editable runtime settings for Streamlit and the active terminal workflow. Secrets stay in local `.env` or the process environment. Process environment values and legacy `.env` settings override the TOML values.
- Code requires claim support, validates span membership and creates own-turn citation links. It withholds malformed answers and unchecked drafts; empty retrieval uses no generation request.
- Citation provenance is not proof of semantic correctness. Evaluate support and answer quality separately.
- Recent user questions help resolve follow-ups. Earlier assistant answers are not evidence.
- Gemini's process-local rolling allowance defaults to 12 requests per 60 seconds. At capacity, it is skipped if a backup is configured or queued if Gemini is the only provider. All providers share cooldowns across sessions in one process after HTTP 429. Gemini HTTP 402 for depleted prepaid credits also triggers failover. Provider retry hints set the cooldown, with defaults of 60 seconds for HTTP 429 and five minutes for HTTP 402.
- Generation request timeouts and HTTP 408 and 504 trigger immediate failover for the current question. Each later question starts from the configured provider priority and honors any active quota cooldowns. Other generation failures stop the turn.

## Interfaces and experiments

The active terminal sample is `python -m src.rag_study_buddy.gemini_agent`. The legacy `agent.py`/`rag.py` modules retain the four-subagent Ollama graph for regression tests and comparison. The local-model experiment is recorded in the history of `feat-ollama`.

Python 3.13 is the tested runtime. CI runs offline tests and dependency checks on Ubuntu and Windows; live published measurements were performed on Windows. Never require a real API key or model server for offline tests.

## Validation evidence

- Final saved-index single-request burst: five independent service sessions, five Gemini HTTP requests, five local query embeddings, zero retries/429/timeouts and 21 valid own-session citation links.
- Per-question latency: 2.03-2.95 seconds in the final burst and 8.93-15.47 seconds in an earlier successful burst. Neither is a production latency guarantee.
- Final chat usage: 9,038 input + 1,239 output = 10,277 tokens; provider and callback totals matched.
- The 321-chunk local index was reused without corpus re-embedding. Preparation retrieval probes were measured separately.
- Offline tests cover indexing, generation, citation rejection, UI behavior, isolation and legacy graph regressions. Mocked HTTP tests verify quota and timeout failover, shared cooldowns and REST request formats. Published live measurements used Gemini alone.
- Detailed outcomes, including failures and limitations, remain in the validation reports. Raw traces, models and indexes stay in ignored local directories.

## Maintenance rules

Keep source metadata and evidence visible, do not publish unchecked drafts, and do not add silent model calls to repair output. Benchmark changes with real HTTP counts when request usage changes. Record only tests actually run. Use a public GitHub noreply identity for new commits and keep machine-specific paths and conversational logs out of shared documentation.

## Portfolio maintenance

The default main branch contains the multi-provider web workflow, fresh-checkout setup instructions, updated changelog and pinned-action Python 3.13 CI for Ubuntu and Windows. Current shared documentation omits machine-specific paths and internal conversational notes. Existing commit and tag history is intentionally retained. Older author identities remain visible in that history. New local commits use the GitHub noreply identity. Repository visibility remains private.
