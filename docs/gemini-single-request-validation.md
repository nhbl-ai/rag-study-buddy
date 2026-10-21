## Architecture and scope

Date: 2026-10-20 (Asia/Bangkok). Branch: `feat-gemini-single-request`, created from `feat-ollama` at `d149d5d`. The original source changes were isolated on a feature branch before portfolio integration.

The active web entry point now imports `gemini_rag.py`: local retrieval -> one native structured Gemini generation -> code validation -> code-built citation links. There are no evidence-analyst subagents, tool selection loops, LLM-based validators or citation-repair model calls. Gemini retry count is zero; SDK automatic function calling is explicitly disabled. No retrieved spans means zero generation requests; model/provider/output failures do not trigger an automatic second attempt.

Gemini: `gemini-3.5-flash-lite`, max output 2048, 60s request timeout, provider-default sampling. Embeddings: local `embeddinggemma:300m`, CPU requested with num_gpu=0; the runtime after final validation reported zero embedding VRAM. The existing local index contained 321 chunks with 768-dimensional vectors. The benchmark verified its content/model-digest/endpoint cache identity before use, without downloading or re-embedding the corpus. The old Gemini agentic tests used a different 322-chunk/3072-dimensional index; this is an architecture experiment, not an isolated provider or corpus comparison.

Five separate prepared RagSession objects had empty histories and independent model/embedding clients and citation anchors. They asked the same five different questions used in the earlier concurrency study, released by a barrier. These are concurrent RAG service sessions, not five actual browser sessions or a deployed-network capacity test. Offline Streamlit AppTest checks cover UI wiring separately.

A process-local rolling admission budget defaults to 12 generation requests/60s and is shared across sessions and keys with consistent model/quota-group settings. It admits an initial five-request burst rather than serializing all five questions. Other processes, external project traffic, TPM and daily quotas are not coordinated by this in-memory limiter.

## Measurements

The harness wraps each SDK HTTP client's send boundary and separately counts SDK invocations/callbacks. It stores no headers, URL query strings, keys or Settings serialization. Token counts use final cumulative provider metadata and match LangChain callback totals for every user. Embedding/token-generation requests are counted separately. Preparation performs five additional local retrieval probes; those are excluded from per-question request counts and latency. Preparation/model loading and Python launch are excluded; each user's latency includes question retrieval, admission, generation and citation validation.

The initial run passed all five workflows, with no HTTP 429 or retries, in 15.475s overall. Individual seconds were 11.222, 9.887, 12.768, 15.474 and 8.929; 17 citation links; 8,918 input + 1,209 output = 10,127 chat tokens. It requested temperature=0, which the installed provider adapter explicitly ignored for this model. Final code removes that unsupported override, explicitly disables SDK automatic function calling, and asks for focused claims with numeric/identifier details supported by cited spans. The final rerun below checks the actual final source. The prompt and client-config differences plus remote latency variation mean the speed difference cannot be attributed to a single change. Both runs are retained.

### Final five-user run

| User | Question topic | Seconds | Gemini HTTP | Local embedding HTTP | Citation links | Input tokens | Output tokens | Total tokens | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Subagent definition/use | 2.784 | 1 | 1 | 6 | 1,661 | 403 | 2,064 | PASS |
| 2 | Context bloat | 2.453 | 1 | 1 | 4 | 2,205 | 270 | 2,475 | PASS |
| 3 | When to avoid subagents | 2.031 | 1 | 1 | 3 | 1,645 | 180 | 1,825 | PASS |
| 4 | CompiledSubAgent vs dictionary | 2.951 | 1 | 1 | 4 | 1,871 | 196 | 2,067 | PASS |
| 5 | Streaming/subagent activity | 2.695 | 1 | 1 | 4 | 1,656 | 190 | 1,846 | PASS |

Final batch: **2.952s**; mean user latency **2.583s**; start skew **1.439ms**. Peak observed overlapping generation HTTP sends: **5**. Admission waiting was below 0.01ms per user. All five generation HTTP requests returned 200; exactly one SDK/callback/HTTP generation request per user; no retries, 429, timeout or tool calls. All five question embedding HTTP calls returned 200 locally.

Chat usage: **9,038 input + 1,239 output = 10,277 total tokens**. Thinking and cached-content metadata were both zero. There were **21** citation-link occurrences, all resolving to the corresponding user's actual evidence. Citation occurrences are not distinct source count.

### Separate retrieval

| User/question | Preparation retrieval seconds | Retrieved chunk IDs |
|---:|---:|---|
| 1 | 0.035 | 108, 105, 104, 103 |
| 2 | 0.030 | 240, 103, 209, 105 |
| 3 | 0.027 | 105, 108, 104, 137 |
| 4 | 0.035 | 108, 115, 105, 136 |
| 5 | 0.036 | 147, 245, 148, 244 |

The primary probe retrieved source passages defining subagents, explaining context quarantine and listing when to use/avoid them. The comparison probe included dictionary configuration and prebuilt LangGraph/CompiledSubAgent text. The streaming probe included coordinator/subagent streaming and separate event projections. Retrieved passages also contain tangential configuration/code/navigation material; these spot checks are not precision/recall scores. The first initial-run probe took 1.674s; subsequent initial probes took 0.029-0.037s. Final probes were warm.

## Citation and answer review

For each returned claim, code validates a required support list, a quote ID in the supplied schema, exact membership of the corresponding span in retrieved source text, and the absence of model-written citation links/URLs/HTML in claim text. It then constructs links using that turn's evidence anchors. Raw model citation URLs are not trusted. Validation failure prevents any complete answer or unchecked draft from being published.

Manual review of the final five answers and their selected source spans found the main claims consistent with those passages, including the CompiledSubAgent comparison that was misleading in an earlier small-model test. The answers are concise; the streaming response identifies relevant APIs rather than providing a runnable example. This is a five-question spot check, not a gold-standard factual correctness score or proof of semantic entailment. The model's insufficient=false flag is reported as fully_answered by the harness and is not independent evidence of complete coverage. In the initial run, a 500-word detail was present in the retrieved passage but not in the chosen span IDs; this illustrates why ID/anchor validation alone cannot guarantee that every detail is supported by its selected quote.

## Checks and reproducibility

- All **46 offline tests passed** after the final code change: single-call generation, automatic-function-calling disablement, withholding invalid/forged/unbacked output without model repair, abstention, concurrent anchor isolation, key-safe errors, local embedding-only preflight/cache compatibility, shared admission, UI regressions and the retained legacy Ollama graph.
- `pip check`: no broken requirements. `git diff --check`: clean.
- Raw initial results: ignored `work/gemini-single-five/`; final results: ignored `work/gemini-single-five-final/`. These contain actual usage, HTTP timing/counts, public source evidence and per-user answers, not credentials.
- Reproduce using `scripts/benchmark_gemini_single.py` and a verified matching local index, with a fresh output directory. See README for commands. It makes real API calls.

The saved-index experiment demonstrates five successful simultaneous questions with one Gemini generation each on this project/model/account at the measured time. It does not establish a maximum supported user count, production reliability, performance on every machine, a steady 3-second latency, or quota availability under other traffic. API latency varied substantially between the two successful bursts. No billing/quota settings, .env, normal server settings, private documents or original branch references were changed. No corpus embedding cost or cloud GPU is required for this measured path.
