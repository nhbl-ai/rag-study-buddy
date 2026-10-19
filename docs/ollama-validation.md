# Ollama migration validation

Date: 2026-10-19. Branch: `feat-ollama`, based on `feat-chat-ui` commit `480fe47`.

## Scope

This is a local-backend evaluation baseline. The original coordinator, `search_documents` retrieval, `task` delegation to evidence-analyst, evidence-file reading, streaming and citations remain. Prompts and graph construction are retained. A new completion check rejects a response that skipped retrieval or skipped delegation after retrieving passages; it does not prove grounding or citation completeness.

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
