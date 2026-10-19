# Ollama migration validation

Date: 2026-10-19. Branch: `feat-ollama`, based on `feat-chat-ui` commit `480fe47`.

## Scope

The sections before **Enforced workflow CPU-only rerun** record the original local-backend evaluation baseline. At that point, the original coordinator, `search_documents` retrieval, `task` delegation to evidence-analyst, evidence-file reading, streaming and citations remained. Prompts and graph construction were retained. A new completion check rejects a response that skipped retrieval or skipped delegation after retrieving passages; it does not prove grounding or citation completeness.

## Environment

- Windows 11; Intel i7-13650HX; 16 GB RAM; RTX 4050 Laptop with approximately 6 GiB VRAM.
- Ollama 0.40.5; langchain-ollama 1.1.0; Python Ollama SDK 0.6.3.
- Default chat candidate: `qwen3.5:4b`; embeddings: `embeddinggemma:300m`.
- Chat context: 8192; request timeout: 180 seconds; one parallel inference slot.
- Localhost only, cloud features disabled, model storage under ignored `work/ollama-models/`.
- Existing Gemini credentials and indexes were not displayed, overwritten, or sent to a model service.

## Checks

| Check | Result |
|---|---|
| Offline regressions | 22 passed: cache/resume/invalidation, metadata, real graph delegation and read_file with a fake model, stream filtering, UI, terminal, safe failures and skipped-flow detection |
| Dependencies | `pip check` passed |
| Source syntax and whitespace | Python compilation and `git diff --check` passed |
| Runtime and UI | Ollama local-only startup confirmed; Streamlit `/_stcore/health` returned `200 ok` |
| Live document embeddings | 321 chunks indexed locally; initial preparation about 93 seconds including cold embedding load and downloads |
| Cache reuse | 321 saved passages reused; about 5 seconds including fresh source downloads |
| Retrieval, checked separately | Four results for the sample question came from relevant subagent documentation, preserving URLs and chunk IDs |
| End-to-end local generation | Not yet passed the required retrieval/delegation/file-reading sequence |
| Concurrent-user load | Not tested |

## Generation observations

The sample question is: "What is a subagent, and when should I use one?"

Qwen3 4B instruct, Qwen3 4B and Qwen3 8B trials returned an answer without calling document search. Qwen3.5 4B called search and read evidence files directly on the coordinator, skipping the required evidence-analyst subagent. These responses are not accepted as successful RAG results. A direct isolated ChatOllama tool-call probe succeeded; actual HTTP payload inspection confirmed the original system prompt and task schema were transmitted.

An additional streaming question with Qwen3.5 4B also retrieved and read evidence on the coordinator rather than delegating. Model capability checks therefore establish API support, not reliable compliance. The 8B trial also spilled onto CPU (observed 37% CPU / 63% GPU in `ollama ps`). Qwen3.5 4B and the embedding model were observed fully GPU-loaded together. Sampling now follows model defaults; Qwen advises against greedy decoding in its [model documentation](https://huggingface.co/Qwen/Qwen3-8B).

This evaluation does not establish answer quality, citation completeness, or capacity for concurrent users. Source downloads still require Internet access. Local diagnostics and generated indexes remain ignored by Git.

## Integrated GPU and CPU comparison

The user requested a test without the discrete GPU. A separate server on localhost:11435 enabled Intel UHD through Vulkan, using `OLLAMA_IGPU_ENABLE=1`, `OLLAMA_VULKAN=1`, `GGML_VK_VISIBLE_DEVICES=0`, and `CUDA_VISIBLE_DEVICES=-1`. Vulkan index 0 was verified as Intel RaptorLake-S Mobile Graphics Controller on this machine; indices must be verified on another machine. Intel driver: 32.0.101.7026. The main server configuration was not changed.

The Intel device was detected as an iGPU. Logs confirmed all 33/33 Qwen3.5 layers offloaded to Intel Vulkan; `/api/ps` reported the full loaded model size in `size_vram`. RTX utilization stayed at 0% during that test. The approximately 7.9 GiB reported for Intel is shared system memory, not dedicated VRAM.

All chat trials used Qwen3.5 4B Q4_K_M, an 8192-token context, the same 80-token evidence prompt, thinking disabled, model-default sampling, and an output limit of 96 tokens. The table records one warm request per device, immediately following a cold request. Prefix caching can help these repeated prompts; output lengths vary because sampling is not fixed. Tokens/second is decode speed; wall time includes request/prefill overhead. This is a short inference probe, not an end-to-end RAG or load benchmark.

| Device | Decode tokens/s | Warm request seconds | Output tokens |
|---|---:|---:|---:|
| RTX 4050 Laptop | 49.09 | 1.35 | 61 |
| i7-13650HX CPU-only | 10.20 | 6.15 | 61 |
| Intel UHD via Vulkan | 3.72 | 19.35 | 69 |

Cold chat requests took about 31.23 seconds on Intel, 11.99 seconds on CPU, and 7.81 seconds on RTX, including model loading. CPU-only was explicitly selected with `num_gpu=0`; `/api/ps` reported `size_vram=0`. Models were unloaded between device comparisons.

EmbeddingGemma also worked on Intel: the first single-vector request took 7.56 seconds including load; a warm four-passage batch took 0.48 seconds and returned four 768-dimensional vectors. These were short synthetic public passages, not the full document corpus.

The lowest sampled free physical RAM during the Intel chat cold request was about 0.89 GiB. RAM measurements reflect other applications running on the machine as well. CPU-only was roughly 2.7 times faster than Intel UHD in this probe; the result must not be generalized to every CPU or integrated GPU. On a similar machine without a discrete GPU, CPU-only is a practical starting point. The existing agent/delegation quality limitations remain separate from hardware feasibility.

The temporary iGPU server was stopped after testing; the application remains on the normal localhost:11434 server. Raw logs and measurements are kept in ignored `work/igpu-bench/`. No drivers or global GPU settings were changed. [Ollama hardware documentation](https://docs.ollama.com/gpu) describes Vulkan device selection.

## Full RAG on CPU-only

On 2026-10-19, the user requested a full CPU-only RAG run, including subagent and citation checks. A separate server at localhost:11436 disabled both CUDA and Vulkan. Both EmbeddingGemma and Qwen3.5 were observed with `size_vram=0`; the main server configuration was unchanged.

The test used the existing question, "What is a subagent, and when should I use one?", Qwen3.5 4B, EmbeddingGemma 300M, context 8192, timeout 180 seconds, empty history, model-default sampling and thinking. Thinking was not disabled as in the short hardware probes. Prompts, graph, tools and completion guard were unchanged; callbacks observed real tool execution and nested graph namespaces. A fresh ignored test cache ensured all 321 chunks were embedded on CPU.

| Phase | Measured seconds |
|---|---:|
| Cold preparation: downloads, splitting and indexing 321 chunks | 64.77 |
| Separate retrieval probe, top four passages | 0.035 |
| RAG question processing until application rejection | 156.89 |
| Preparation with the matching saved index, including downloads | 2.56 |

| Required step | Observed |
|---|---|
| `search_documents` | 1 successful execution; four passages retained URLs and chunk IDs |
| `task` to evidence-analyst | 0 executions |
| Evidence-analyst model calls | 0 |
| `read_file` | 4 successful executions, all in the coordinator namespace |
| Numbered citation links in candidate answer | 0 |
| Accepted application answer | No; `RagFlowError` rejected the skipped delegation |

There were three coordinator model calls, taking approximately 49.38, 39.46 and 67.78 seconds. The candidate answer explained subagents but contained no citation links or source URLs. It was a rejected candidate, not a completed RAG answer. This is a failed workflow/citation check, even though indexing, retrieval and CPU inference executed successfully.

The lowest sampled free physical RAM was 0.90 GiB; other running applications affect this value. No successful full-flow response time can be claimed from this run: 156.89 seconds is time to rejection and includes no evidence-analysis subagent work. The result is one question, not a reliability or concurrent-load evaluation. Raw measurements, callback trace, candidate text and retrieved passages are in ignored `work/cpu-rag-test/`. The temporary server was stopped after testing; application source and its normal GPU settings were not modified.


## Enforced workflow CPU-only rerun

The user authorized fixing orchestration on `feat-ollama` after the baseline above. Code now requires `search_documents`, then one real Deep Agents `task` per distinct retrieved passage. Each task runs a compiled isolated `evidence-analyst` LangGraph subgraph with a mandatory `read_file -> analyze` path. Jobs run sequentially; each subagent receives only its assigned in-memory file.

The analyst selects numbered spans from the actual file; code extracts those spans verbatim. A per-file JSON schema enumerates allowed span IDs. Synthesis returns structured claims referencing unique quote IDs, such as e2q1, constrained by a schema enum over supplied quotes. Code maps those IDs back to evidence and renders canonical per-turn citations. Invalid output, unknown IDs, missing support or model-written citation links fail closed. The UI publishes only the validated answer, retaining progress/evidence events. Empty or irrelevant evidence produces an explicit insufficient-information answer. These checks establish provenance and citation structure, not proof of semantic entailment.

Chat remains Qwen3.5 4B, embeddings remain EmbeddingGemma 300M, retrieval remains top four, and context remains 8192. Chat now uses `reasoning=False`, `num_predict=1024`, native JSON-schema output and otherwise default sampling. Prompts and orchestration changed; timings are not a controlled model-speed comparison against the original failure.

### Retained iterations

- `work/cpu-rag-fixed/`: fresh indexing 67.10s; rejected a model-retyped quote mismatch after 34.59s. Actual subagent execution succeeded; no answer was accepted. Source-span selection replaced model quote copying.
- `work/cpu-rag-enforced/`: fresh CPU indexing of 321 chunks 68.23s; separate warm retrieval 0.077s; structurally accepted answer 71.14s. Manual review found a definition confusing a configuration dictionary with the agent concept. A UI follow-up succeeded; an unrelated question was safely rejected for span ID 0. Clearer prompts and per-file schema enums followed.
- `work/cpu-rag-final/`: primary answer accepted in 71.48s with five citation links and an improved definition; unrelated-question abstention passed. A UI follow-up was rejected for referencing quote 4 when only three quotes existed. This led to unique quote IDs instead of separate evidence/quote numbers.
- `work/cpu-rag-final-refs/`: final source and all three live cases passed. The shared index was built entirely on CPU in the preceding fresh-index run; both models were unloaded before the final primary measurement. Raw model outputs, task returns, selected quotes, evidence and tokens remain in ignored local output.

### Final measurement

Sample: "What is a subagent, and when should I use one?" Empty history. Separate localhost:11436 server with CUDA and Vulkan disabled. Both models were observed with `size_vram=0`; normal localhost:11434 configuration was retained.

| Phase/check | Result |
|---|---|
| Fresh CPU preparation, same indexing pipeline, 321 chunks | 68.23s |
| Final preparation from that saved index, including downloads | 5.17s |
| Separate top-four retrieval after unloading embedding model | 1.221s, including cold embedding load |
| Final complete question phase | 84.35s; accepted |
| Four analyst model calls | 9.91s, 6.09s, 6.63s, 5.78s |
| Synthesis model call | 55.83s |
| Repeated saved-index preparation, including downloads | 4.34s |
| search_documents / Deep Agents task / analyst model / nested read_file | 1 / 4 / 4 / 4 successful executions |
| Citation links | 8 links across 8 claim paragraphs; zero unknown links |
| All selected quotes checked against actual file content | Passed |
| Model tokens across five calls | 2494 input + 534 output = 3028 |
| Lowest sampled free physical RAM in final primary run | 1.50 GiB; earlier fresh-index run reached 0.87 GiB |
| Placement | Chat 3,376,404,232 bytes; embedding 692,993,392 bytes; zero VRAM for both |

Retrieval was checked separately: all four results came from the Subagents page, chunks 108, 105, 104 and 103, covering configuration, when to delegate, context bloat and delegation. Manual review checked the final answer against the retrieved passages and selected quotes; the definition no longer treats a subagent itself as a dictionary. This is a small sample, not a retrieval recall or grounding benchmark.

### Streamlit with real CPU Ollama

AppTest used a real prepared RagSession and the same CPU-built index. The completed primary turn seeded the history for the follow-up. The interface exercised actual model calls, tools, source metadata, citation controls and evidence selection.

| Case | Result | Seconds |
|---|---|---:|
| Follow-up: When should I use it instead of adding more tools to the main agent? | Passed; two history messages, four tasks/read_file calls, five valid citation links | 72.19 |
| Unrelated question: Which team won the 2026 FIFA World Cup? | Passed; four tasks/read_file calls, explicit insufficient-information answer, no invented team | 41.20 |
| New conversation | Cleared history and retained the real prepared index/service | — |
| Evidence UI | Selected the current turn; citation popovers and passages rendered | — |

Both models remained at zero VRAM in UI checks. Final primary generation, follow-up and abstention all passed; this is three individual cases, not a statistical reliability claim. The final UI abstention used four analyst calls and one synthesis call because an analyst selected a source span but synthesis correctly found insufficient information.

Final offline checks: 24 tests passed in 7.84s, including actual Deep Agents task/file access with a fake provider, stop-before-synthesis failures, invalid IDs/support/links, per-turn anchors, history, cache/resume and UI/terminal behavior. Dependency and compilation checks passed; whitespace checks were clean.

This demonstrates CPU-only operation on the tested i7/16 GB laptop. It does not establish capacity for concurrent users or performance on every machine. A single inference slot and roughly 72–84 seconds per supported test question limit CPU throughput. Public source downloads still require Internet access. The temporary CPU server is stopped after testing and Streamlit is restarted on the normal server.
