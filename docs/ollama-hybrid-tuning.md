# CPU + Intel iGPU tuning and five-user RAG validation

Date: 2026-10-19. Branch: `feat-ollama`. Hardware: i7-13650HX, Intel UHD / RaptorLake-S Mobile Graphics Controller, 16 GB system RAM, Windows 11. Ollama 0.40.5; Qwen3.5 4B Q4_K_M and EmbeddingGemma 300M. RTX 4050 was excluded from the temporary Vulkan server; main Ollama/UI processes were left running.

## Selection and runtime limitation

Four real model probes requested 2, 8, 16 and 24 of the 33 chat layers on Intel. They reconstructed analyst and synthesis prompts from the previous run's actual retrieved evidence and verified quotes. Each candidate used one slot, context 8192, thinking off, seed 42 and short constrained outputs capped at 128 tokens. These probes select a candidate; they do not execute the application subagents or constitute full RAG validation.

| Intel layers | CPU layers | Analyst prefill, s | Analyst decode, tokens/s | Synthesis prefill, s | Synthesis decode, tokens/s | Weighted service score, s |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 31 | 16.37 | 6.74 | 25.89 | 5.60 | 163.33 |
| 8 | 25 | 14.41 | 6.35 | 21.56 | 5.62 | **151.61** |
| 16 | 17 | 13.97 | 5.48 | 20.57 | 4.93 | 159.10 |
| 24 | 9 | 13.88 | 4.45 | 19.94 | 4.35 | 170.34 |

The heuristic score is `4 * (analyst_prefill + 16 / analyst_decode_rate) + synthesis_prefill + 350 / synthesis_decode_rate`. It normalizes differing generated lengths to an estimated four-analyst/one-synthesis workload. It excludes queueing and does not predict a user's observed latency. Eight Intel layers had the lowest score among these candidates; this is not an exhaustive optimum or a steady-state performance claim.

A subsequent probe requested two slots and dispatched five short synthesis requests. All five completed in 156.40s, but the runtime warned `model architecture does not currently support parallel requests architecture=qwen35` and launched chat with `-np 1`, `n_slots = 1`, `n_ctx_slot = 8192`. Therefore this is **not a two-slot throughput result**. The remaining layer candidate was cancelled before its five-request dispatch, and redundant three/five-slot sweeps were skipped. The model/runtime combination cannot provide multiple inference slots merely by changing `OLLAMA_NUM_PARALLEL`. This limitation agrees with the [upstream scheduler discussion](https://github.com/ollama/ollama/issues/14638). No third-party code was changed to bypass it.

The previous full hybrid run's prompt cache reached approximately 1,405 MiB while free physical RAM dipped to 0.46 GiB. The selected Intel profile bounds that prompt cache to 512 MiB using the installed llama-server's documented CLI environment setting `LLAMA_ARG_CACHE_RAM`. This limits cached prior prompts, not the 8192-token active context. The bound can trade cache reuse for memory headroom; it is not proven to be the sole cause of any performance difference.

## Selected profile and application changes

- Chat: **8/33 layers on Intel**, remaining **25 on CPU**; embedding: **CPU only**.
- One requested and actual inference slot; at most two loaded models; prompt cache limit **512 MiB**.
- Context **8192**, unchanged **180s** client request/read timeout, thinking disabled, output cap **1024**, model-default sampling for the full RAG test.
- `Settings` now accepts optional `OLLAMA_CHAT_GPU_LAYERS` and `OLLAMA_EMBEDDING_GPU_LAYERS`; blank retains automatic placement, zero requests CPU. Both clients receive their own placement. Invalid/negative overrides fail before requests. Hardware placement is excluded from vector-cache identity because it does not change the embedding model/digest or documents.
- `scripts/start_ollama.ps1` exposes `-IntelHybrid`, `-ParallelSlots`, `-Port`, `-VulkanDevice` and `-PromptCacheMiB`. The Intel profile selects Vulkan and excludes CUDA/ROCm/HIP, bounds the prompt cache and permits two loaded models. Default invocation retains the original automatic-device, one-slot behavior.
- A process-local shared semaphore admits **one chat request per endpoint/model** before starting the timed HTTP request. Analyst and synthesis calls share the gate across sessions; exceptions release it. Normalized endpoint trailing slashes and implicit `:latest` tags share the same gate. Request metadata records `inference_slot_wait_seconds`. The 180s timeout remains an HTTP request/read timeout; application queue wait and total RAG time are not bounded by it. Independent app processes do not share this gate.
- No changes to retrieval k=4, isolated Deep Agents task dispatch, mandatory nested read_file, verbatim span selection, dynamic quote-ID schemas, citation validation, abstention behavior, history or UI publication rules. No API keys or `.env` changes.

Commands to use this profile are in [README](../README.md#optional-cpu--intel-igpu-profile). The test uses localhost:11435 and the repository-local model store; it does not change the main server at 11434 or UI at 8501.

## Full five-user test

The same five questions and saved 321-passage index as the [initial hybrid baseline](ollama-concurrency-validation.md) are used. Five independent prepared RagSession instances synchronize submission with a barrier, each with its own store, empty history, evidence files and citation anchors; Ollama shares the model weights. Preparation, cold warmup, document downloads and index construction are excluded from question latency. Retrieval is evaluated separately before dispatch. The full run uses the new Settings fields, not monkey-patching the model client.

### Layer/cache tuning before application admission

The first full tuned run used the selected eight-layer split and 512 MiB cache, without application admission (`c4248ec`). All twenty subagent tasks, analyst calls and nested file reads passed. Only **3/5 responses** were accepted structurally; two synthesis requests timed out before their first token. User times: 486.95s (timeout), 377.07s (five citation links), 419.30s (two links), 498.46s (four links), 472.52s (timeout). Batch duration 498.48s; minimum sampled free RAM 0.662 GiB.

This is not a successful speedup: two clients terminated early, so the shorter batch duration cannot be compared as if all five finished. It justified moving chat admission ahead of the timed HTTP request rather than raising the timeout or removing workflow steps.

Manual review also found a semantic error in this run's structurally accepted user 4 answer: it attributed the isolated-versus-forked distinction in access to the task tool to CompiledSubAgent-versus-dictionary types. Its links and quote IDs existed, but that claim overgeneralized the retrieved table. Structural/citation checks do not catch this reasoning error. The test reports accepted responses separately from semantic correctness.

### Final run with application admission

RAG source revision: `9ffffda`. Runtime logs confirm Intel Vulkan device 0, **8/33 layers offloaded**, **one actual slot**, **8192 context per slot** and effective prompt-cache budget **512 MiB**. Chat `/api/ps` reports 3,998,639,714 bytes total and 1,282,775,448 bytes on the graphics device; Intel graphics allocations use shared system RAM. Embeddings report zero GPU memory.

| User | Total latency including all queue waits, s | Sum of application slot waits, s | Valid citation links | Result |
|---:|---:|---:|---:|---|
| 1 | 544.93 | 391.41 | 5 | Passed workflow/citation checks |
| 2 | 363.07 | 239.37 | 3 | Passed workflow/citation checks |
| 3 | 451.02 | 356.94 | 2 | Passed workflow/citation checks |
| 4 | 641.89 | 480.52 | 6 | Passed workflow; comparison needs semantic revision |
| 5 | 414.07 | 304.11 | 3 | Passed workflow/citation checks |

- **5/5 completed** with accepted structured answers and citations; **zero errors/timeouts** in this burst. This is workflow success, not proof that every claim is correct.
- All five searches, **20 Deep Agents task executions**, **20 analyst calls**, **20 nested read_file executions** and **five synthesis calls** completed. Every selected quote was verbatim in its assigned passage and all **19 citation links** resolved to that user's own evidence anchors.
- Batch duration **641.89s**, approximately **10 minutes 42 seconds**. Mean user latency **482.99s**, median **451.02s**; range **363.07–641.89s** (about 6.1–10.7 minutes). Dispatch skew **0.0073s**. Warmup/preparation is excluded; every application queue wait is included.
- Callback timing intervals show at most **one active chat call**. First internal model token after admission: **9.81–22.25s**; these are private analyst/synthesis tokens, not displayed answer latency. The longest admitted chat call lasted **96.96s**.
- User 4's synthesis waited **235.94s in the application** before opening its HTTP request, then completed. The timeout was not raised: it remains 180s for HTTP request/stream reads and does not bound application queueing or whole-RAG latency. Admission avoids idle HTTP waits while retaining the user's full waiting time in the measurement.
- Lowest sampled free physical RAM **0.474 GiB**, also during the question phase. The 512 MiB prompt-cache budget does not cap overall process memory. The machine remains memory-constrained; no material whole-system RAM improvement can be inferred from the small difference versus the original 0.46 GiB minimum.
- Separate retrieval probes returned the same chunk IDs in the same order as the original hybrid run: **1.361s** for the initial cold embedding query and **0.029–0.034s** warm for the remaining probes. Retrieval did not change to obtain these results.
- Chat usage across all 25 measured calls: **12,715 input + 1,551 output = 14,266 tokens**. Preparation, standalone retrieval probes and warmup are outside this chat usage total.
- **Manual semantic limitation:** user 4's answer still gives a misleading comparison about fork mode and dictionary configuration. Its citations are structurally valid, but existence of a quote ID does not establish the claimed distinction. This answer needs revision; the run must not be presented as five semantically verified answers. Other answers were spot-checked, not evaluated with a comprehensive quality benchmark.

### Comparison with the previous attempts

| Configuration | Accepted cited responses | Synthesis timeouts | Batch duration, s | Minimum free physical RAM, GiB |
|---|---:|---:|---:|---:|
| Initial 16 Intel layers, one slot, default prompt cache | 4/5 (one partial) | 1 | 541.01 | 0.460 |
| Tuned 8 Intel layers, 512 MiB cache, before admission | 3/5 | 2 | 498.48 | 0.662 |
| Tuned 8 Intel layers, 512 MiB cache, application admission | **5/5** | **0** | **641.89** | **0.474** |

The final run completed more work instead of timing out clients. Its longer batch duration is not evidence that admission increased model speed or that tuning achieved a speedup. Different sampled output lengths and failed requests also prevent a clean causal speed comparison. Admission improved completion in this observed burst; hardware throughput remains limited to one real slot. The selected configuration supports a demonstration or a small queued workload on this laptop, but these multi-minute latencies are unsuitable evidence for responsive service to many simultaneous users or general feasibility across all machines.

Both temporary models were unloaded and the temporary server/launcher exited; ports 11435/11436 are free. Main Ollama 11434 and Streamlit 8501 returned HTTP200 after cleanup. Main application process settings were not switched to the experimental profile.


## Validation and limitations

All **28 offline regression tests** passed, including actual Deep Agents subgraph/read_file execution with a fake provider, citation rejection, UI behavior, placement overrides, shared admission across sessions, provider calls blocked until a slot opens, release on errors, request metadata isolation, and preserving the embedding cache when only device placement changes. Dependency check, Python compilation, PowerShell parser check and Git whitespace check passed. The live test checks real model execution independently from these offline tests.

A single five-question burst on this laptop does not establish reliability, capacity across different machines or answer quality across a wider corpus. Valid citations prove source mapping and exact selected quotations, not semantic entailment for every claim. Question 4's retrieved chunks contain CompiledSubAgent discussion but not necessarily enough selected evidence for the full requested comparison; explicit partial answers must be reported separately. Default sampling and differing output lengths also limit direct timing attribution between runs. No five-browser/frontend/network load test was performed.

Raw short probes: ignored `work/hybrid-tuning/slots-1/` and `slots-2/`. First tuned run and its harness: ignored `work/hybrid-optimized/hybrid/` and `work/hybrid-optimized/run.py`. Final admission run: ignored `work/hybrid-admission/hybrid/` and `work/hybrid-admission/run.py`. Previous raw baseline remains unchanged in `work/concurrency5/hybrid/`. Logs record device placement, effective slot/context counts and effective prompt-cache bound; results include retrieval scores/text, per-user tool/model traces, selected quotes, answers, errors, token usage and sampled RAM.
