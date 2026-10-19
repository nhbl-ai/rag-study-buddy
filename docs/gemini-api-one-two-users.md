# Gemini API: one- and two-user full-RAG validation

Run window: 2026-10-19 to 2026-10-20 (Asia/Bangkok). Model: `gemini-3.5-flash-lite`; embeddings: `gemini-embedding-001`. Tests used the original API implementation from `feat-chat-ui` at `480fe47` via an ignored source snapshot, while the checked-out branch remained `feat-ollama`. No application sources, `.env`, billing or quotas were changed.

## Method and scope

All question sessions used the same question, "What is a subagent, and when should I use one?", empty histories and independent evidence/anchors. The saved public-documentation index (322 chunks, 3072 dimensions) was reused without document download or corpus re-embedding. Concurrent sessions were released through a barrier. Between successive phases a benchmark-only wait allowed at least 65 seconds after the preceding phase's last SDK generation completed. That preparation/cooldown is excluded from question latency.

The first original-prompt session completed text but ran only three evidence tasks and emitted bare citation numbers, so it failed the required workflow. Subsequent phases added benchmark-only prompt instructions explicitly requiring four separate evidence-analyst task calls, direct reads of supplied paths, and exact Markdown citation links. The underlying original API graph, tools, model, timeout (60 seconds) and max_retries (2) were retained. This is a prompt-constrained validation of the original agent architecture, not evidence that the unchanged application reliably enforces the workflow. The Ollama deterministic five-call architecture was not substituted.

Actual outbound generation/embedding HTTP sends were counted by wrapping each SDK client's in-memory httpx send method, including native SDK retries. Only URL paths, operation, timing and status codes were retained; query strings, headers and credentials were excluded. SDK invocations are separately recorded and may each include more than one HTTP attempt. No third-party files were modified. Chat token totals use the last cumulative SDK usage metadata per invocation, cross-checked against inherited LangChain callbacks without adding cumulative streamed chunks repeatedly. Embedding responses did not expose billing usage.

The unpaced two-user phase was followed by a separate phase admitting generation HTTP attempts at least five seconds apart across both users, including retries. This is a benchmark-only shared rate limiter; requests and subagent work can overlap. Queue waits are included in user latency. It does not serialize whole user questions or change application defaults.

## Measured results

| Phase | User | Seconds | Generation HTTP attempts | HTTP 200 | HTTP 429 | Retry attempts | Embedding HTTP | Successful tasks / nested reads | Citation links | Full workflow |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| Original prompt, 1 user | 1 | 27.70 | 9 | 9 | 0 | 0 | 1 | 3 / 3 | 0 | FAIL |
| Four-task prompt, 1 user | 1 | 10.85 | 11 | 11 | 0 | 0 | 1 | 4 / 4 | 10 | PASS |
| Four-task prompt, 2 users unpaced | 1 | 11.22 | 11 | 9 | 2 | 1 | 1 | 3 / 4 | 0 | FAIL |
| Four-task prompt, 2 users unpaced | 2 | 12.22 | 12 | 10 | 2 | 1 | 1 | 3 / 4 | 0 | FAIL |
| Four-task prompt, 2 users paced | 1 | 107.63 | 11 | 11 | 0 | 0 | 1 | 4 / 4 | 0 | FAIL |
| Four-task prompt, 2 users paced | 2 | 139.99 | 16 | 16 | 0 | 0 | 1 | 4 / 8 | 4 | PASS |

A successful task count is lower than launched tasks when a subagent terminates with a rate-limit error. Read-file completion alone does not imply successful analysis, synthesis or citations. Citation counts are link occurrences, not distinct source count. A failed row records partial attempts only and must not be treated as the cost of a completed chat.

### SDK generation breakdown

| Phase | User | Coordinator invocations | Analyst invocations | Total SDK invocations |
|---|---:|---:|---:|---:|
| Original prompt, 1 user | 1 | 3 | 6 | 9 |
| Four-task prompt, 1 user | 1 | 3 | 8 | 11 |
| Four-task prompt, 2 users unpaced | 1 | 2 | 8 | 10 |
| Four-task prompt, 2 users unpaced | 2 | 2 | 9 | 11 |
| Four-task prompt, 2 users paced | 1 | 3 | 8 | 11 |
| Four-task prompt, 2 users paced | 2 | 3 | 13 | 16 |

### Token metadata

| Phase | User | Reported input tokens | Reported output tokens | Reported total tokens |
|---|---:|---:|---:|---:|
| Original prompt, 1 user | 1 | 22,968 | 1,305 | 24,273 |
| Four-task prompt, 1 user | 1 | 29,144 | 2,322 | 31,466 |
| Four-task prompt, 2 users unpaced | 1 | 21,710 | 1,594 | 23,304 |
| Four-task prompt, 2 users unpaced | 2 | 24,196 | 1,241 | 25,437 |
| Four-task prompt, 2 users paced | 1 | 29,171 | 2,199 | 31,370 |
| Four-task prompt, 2 users paced | 2 | 46,580 | 2,915 | 49,495 |

These totals exclude embedding tokens and any failed requests without usage metadata. Failed generation attempts are not assumed to use zero tokens. Preparation retrieval queries are separate embedding calls and excluded from the per-chat request table. No new corpus-indexing token cost is included.

## Retrieval and generation checks

Retrieval was evaluated separately before each phase with a real embedding query; no chat generation is needed for that probe. The saved matches include passages defining subagents/context quarantine and discussing use cases, alongside less directly relevant configuration/forking passages. Question-time search is model-selected, so its query and retrieved chunks can differ from the preparation probe. This is a relevance spot check, not a recall or accuracy benchmark.

- Original prompt, 1 user: separate retrieval 0.699s; batch question duration 27.697s; start skew 0.000ms.
- Four-task prompt, 1 user: separate retrieval 0.757s; batch question duration 10.848s; start skew 0.000ms.
- Four-task prompt, 2 users unpaced: separate retrieval 0.723s; batch question duration 12.225s; start skew 2.727ms.
- Four-task prompt, 2 users paced: separate retrieval 0.745s; batch question duration 139.993s; start skew 3.727ms.

Full-workflow acceptance requires completed final text, retrieval, four successful evidence-analyst tasks, nested read_file coverage of every retrieved evidence file, and citation anchors resolving to that session's actual evidence. These checks establish workflow and citation provenance; they do not prove semantic entailment or quality across arbitrary questions.

## Interpretation

The user reported limits of 15 RPM, 250,000 input TPM and 500 RPD. The unpaced two-user burst hit 429; this is evidence that this particular burst failed, not a universal hard concurrent-user cap. Successful single- and paced two-user rows establish only the measured question/session behavior. A shared limiter can trade waiting time for completion. No throughput, latency or daily-question guarantee follows from one burst, and external traffic/retries share project allowances. See [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Raw inputs, trace, answers and results are under ignored `work/gemini-one-api/`, `work/gemini-one-api-strict/`, `work/gemini-two-api/` and `work/gemini-two-api-paced/`. Credentials, vector stores and large logs must remain uncommitted. Earlier five-user results are in [gemini-api-concurrency-validation.md](gemini-api-concurrency-validation.md).

## Observed capacity and limitations

The four-task single-user phase passed in 10.85s with exactly 11 generation HTTP requests (three coordinator, eight analyst), one question embedding request, no retry and ten valid citation-link occurrences. The original prompt failed structural requirements despite returning text in 27.70s with nine generation requests and one embedding.

Two unpaced users both failed with 429: user 1 issued 11 generation HTTP attempts (nine 200, two 429, including one retry); user 2 issued 12 (ten 200, two 429, including one retry). Each also issued one successful embedding request. These are incomplete-question counts.

With shared five-second generation pacing, both returned text without 429 or retries: user 1 took 107.63s, issued 11 generation plus one embedding and completed four tasks/reads, but returned bare citations without links, so failed. User 2 took 139.99s, issued 16 generation plus one embedding, completed four tasks/eight nested reads, and supplied four valid citation links in a source list. Its body still uses bare reference numbers tied to that list; this is not the same as inline link placement on every claim. It passed the stated workflow/link provenance checks. Its extra five analyst calls came from additional reads/tool loops, not HTTP retries.

Thus generation use for the measured full-workflow successes was 11 and 16 requests per chat, with one embedding each. It is not a fixed universal range. One passing single-user session and one of two passing paced sessions do not establish stable capacity even for one user across arbitrary questions. The unmodified API application has no demonstrated guarantee of four subagents and citation links. Prompt additions alone remained insufficient in one paced session. The request limiter resolved 429 in this burst, but did not resolve citation compliance. No additional load tests or app changes were made after these results.
