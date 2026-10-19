# Gemini API: five simultaneous sessions and quota limits

Date: 2026-10-19. Evaluated application: original `feat-chat-ui` API source at `480fe47`; the Ollama implementation was not substituted into this test. The source snapshot and benchmark files remain in ignored `work/gemini-five/`. The checked-out branch stayed `feat-ollama`; application sources, credentials and both original branch references were left unchanged.

## Scope and measurement

Five independent original RagSession instances submitted the same five documentation questions used in the Ollama concurrency tests, with empty histories and separate evidence anchors. Chat model: `gemini-3.5-flash-lite`; embeddings: `gemini-embedding-001`; original chat timeout 60s and max_retries 2 retained. Model metadata confirmed access before generation. The saved Gemini index contains 322 passages with 3072-dimensional vectors from the three public documentation pages. It differs from the 321-passage Ollama snapshot and was reused without embedding the document corpus again.

A separate retrieval query completed in 0.692s before dispatch. All five question sessions then used real remote embedding queries, tool calls and generation. Original API agents choose their own retrieval/delegation/file-reading calls; this is not the deterministic four-analyst Ollama implementation, so token counts across these versions are not an isolated comparison of providers/tokenizers.

Generation accounting captures the final cumulative `usage_metadata` from each SDK generation invocation. Cumulative chunks are overwritten rather than summed. Per-user LangChain callback input/output/total counts matched these provider totals exactly. Reasoning tokens, when present, belong in output totals; this run reported zero thinking/cached-content tokens. Failed requests without usage metadata are recorded as unknown, not assumed to cost zero. SDK-internal HTTP retry attempts were not separately counted.

## Unpaced simultaneous burst

| User | Reported input tokens | Reported output tokens | Reported total tokens | Elapsed seconds | Outcome |
|---:|---:|---:|---:|---:|---|
| 1 | 9,747 | 99 | 9,846 | 28.12 | 429 before completion |
| 2 | 12,182 | 545 | 12,727 | 38.38 | 429 before completion |
| 3 | 5,608 | 69 | 5,677 | 26.81 | 429 before completion |
| 4 | 2,607 | 24 | 2,631 | 26.62 | 429 before completion |
| 5 | 9,925 | 317 | 10,242 | 46.57 | 429 before completion |
| **Total** | **40,069** | **1,054** | **41,123** | **46.57 batch duration** | **0/5 completed** |

There were 21 observed SDK generation invocations: 16 returned usage metadata, and five ended with 429 errors and no usage metadata. All five user sessions terminated with GoogleRateLimitError/HTTP 429 before a final answer. No final citation links were returned. Therefore **41,123 is observed usage from an incomplete burst, not the token requirement for five complete RAG answers**, and 0/5 completion is not evidence of a measured maximum of four or fewer supported users.

Embedding responses returned vectors but no token statistics or billing metadata. There were five successful question-phase embedding requests plus one separate preparation retrieval query. Their token usage is not included in the 41,123 chat total. A separate countTokens diagnostic reported 14 tokens for the exact primary-question text with the embedding model; that is a tokenization measurement, not embedding billing usage. No corpus-indexing token cost was incurred by this cached-index test.

## Interrupted paced follow-up

A second harness kept the same source/model/prompts but paced generation admission to one new SDK invocation per 10s, with bounded backoff after 429. Before completion, the user clarified that they wanted the user capacity supported by Gemini's quota, rather than a paced token-completion benchmark. Both Python processes belonging to this harness were stopped; no other application/server was stopped. The last saved partial snapshot is archived as `work/gemini-five-queued/interrupted.json`. It is not a completed five-user result or a final consumption total; in-flight requests may have consumed tokens that were not saved before interruption. Consequently the first burst's 41,123 must not be represented as total usage across both attempts.

## Capacity interpretation

Gemini quotas constrain API requests per minute (RPM), input tokens per minute (TPM), and requests per day (RPD), at project level rather than separately per API key. Active limits depend on model and project tier; see [Google's rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

One RAG question can generate multiple coordinator/subagent requests, and each can resend instructions, evidence and prior tool messages. Token allocation alone does not specify a simultaneous-user limit. An approximate steady throughput bound is `min(RPM / average_requests_per_question, TPM / average_input_tokens_per_question)` questions/minute, with a separate daily bound from RPD. This does not directly determine simultaneous users: burst timing, response duration, queueing, retries and shared project traffic matter too.

The user subsequently reported AI Studio readings of 16/15 RPM, 40.87k/250k input TPM, and 47/500 RPD. These identify request rate as an active bottleneck in the reported snapshot: RPM exceeded its limit while input TPM was about 16.35% of its limit and daily requests about 9.4%. The snapshot is not necessarily the exact time window of the first burst, and its usage must not be equated with that burst's saved totals.

The five-user unpaced burst hit 429 and completed no answers. The maximum simultaneous-user count remains unmeasured. With an assumed five generation requests per complete question, 15 RPM and 500 RPD imply theoretical request-only bounds of three questions/minute and 100 questions/day. At ten requests/question, these become 1.5 questions/minute and 50 questions/day. These are conditional examples, not measured full-question costs; the original agent flow can make additional calls. Token limits, retries, other project traffic and response latency may reduce capacity. Five users can be admitted into an application queue with a shared request-rate limiter, but responsive five-user completion has not been validated. No additional generation is needed to explain these limits. Billing, quotas and keys were not changed.


### Per-question call structure

The saved burst recorded 5, 6, 3, 2 and 5 SDK generation invocations per user respectively, including each user's terminal 429 invocation. Each user also issued one successful embedding query. None completed, so these counts are not full-question averages; SDK-internal HTTP retries are not included as separate invocations.

For the intended original API workflow with four evidence files each delegated to a separate analyst, a simple path uses three coordinator calls if task dispatches are batched (search selection, task dispatch, final answer) or six if dispatched one at a time. Each of four analysts needs a call selecting read_file and a call returning its analysis, giving eight analyst calls. This yields a structural estimate of 11-14 generation calls plus one embedding query, assuming no additional tool loops/retrieval/retries. It is not a measured average or fixed bound: the original agent can combine files into fewer tasks, add glob/planning/tool calls or skip steps. The deterministic Ollama workflow's five generation calls must not be assigned to the original API implementation.

At 11-14 generation calls per completed question, 15 RPM implies about 1.07-1.36 questions/minute as a conditional request-only throughput ceiling, and 500 RPD implies about 35-45 questions/day. Two such questions consume 22-28 calls in total; whether they exceed a minute's allowance depends on their timing. These figures do not establish a hard simultaneous-user cap. Five unpaced sessions failed; a successful one- or two-user full-flow baseline has not been measured by this benchmark.


Subsequent measured one- and two-user tests, including actual HTTP retries, four-task prompt validation and shared request pacing, are recorded in [gemini-api-one-two-users.md](gemini-api-one-two-users.md). Successful full-flow rows used 11 and 16 generation requests plus one embedding each; original/unpaced/paced citation failures are retained rather than counted as full successes.
