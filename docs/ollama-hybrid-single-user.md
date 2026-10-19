# One user on CPU + Intel iGPU

Date: 2026-10-19. Branch: `feat-ollama`. RAG source revision: `9ffffda`, unchanged for this measurement.

## Scope and configuration

One real prepared RagSession answered **"What is a subagent, and when should I use one?"**, with empty history and no other user requests on its Ollama endpoint. The benchmark exercised the existing retrieval, real Deep Agents task/file-reading subgraphs, structured evidence selection, synthesis and canonical citation checks. This was a backend measurement, not a timed browser interaction.

Hardware: Intel i7-13650HX, 16 GB RAM, Intel UHD / RaptorLake-S Mobile Graphics Controller. Ollama 0.40.5 ran on temporary localhost:11435, with cloud disabled and CUDA/ROCm/HIP excluded. Runtime logs verify Intel Vulkan device 0 and **8/33 chat layers offloaded**, leaving **25 on CPU**. Embeddings ran on CPU with zero GPU memory. Chat placement reported 3,998,639,714 bytes total and 1,282,775,448 bytes on the graphics device; Intel graphics memory is shared system RAM.

The profile matches the [five-user tuning run](ollama-hybrid-tuning.md): Qwen3.5 4B Q4_K_M, EmbeddingGemma 300M, chat context 8192, one real inference slot, process-local admission capacity 1, prompt-cache budget 512 MiB, timeout 180s, thinking disabled, output cap 1024 and model-default sampling. No prompts, application code, .env or main-server settings were changed.

The existing 321-passage index snapshot was loaded directly. Preparation below includes model preflight, loading the saved index, a separate retrieval probe, chat warmup and session construction. It excludes Python import/server launch, model downloads, public document downloads and fresh index construction. Consequently the preparation+answer total is not a first-installation or uncached document-preparation time.

## Measured result

| Stage or check | Result |
|---|---|
| Preparation using the existing index, including model warmup | **12.32s** |
| Separate initial retrieval probe | **1.490s**, included in preparation |
| Retrieval inside the actual question phase | **0.091s** |
| Four analyst model calls combined | **59.29s** |
| Synthesis model call | **59.70s** |
| Complete question phase, including retrieval/tool orchestration/validation | **119.14s**, approximately **1 minute 59 seconds** |
| Preparation plus complete question phase | **131.46s**, approximately **2 minutes 11 seconds** |
| Deep Agents task executions / nested read_file executions | **4/4**, all successful |
| Chat calls | **4 analysts + 1 synthesis**, all successful |
| Citation links | **3**, all point to this turn's actual evidence |
| Selected quotations | All verbatim in their assigned passages |
| Application slot wait, summed across all five calls | **0.000041s**, effectively no queueing |
| Lowest sampled free physical RAM | **0.466 GiB**, includes other applications |
| Chat usage | **2,425 input + 278 output = 2,703 tokens** |
| Errors/timeouts | **None** |

The separate retrieval returned chunk IDs **108, 105, 104, 103**, in the same order as the previous primary-question probe. Retrieval remained distinct from generation evaluation. Internal first-token times after admission ranged from 10.09s to 18.73s. These tokens are private model work; the user receives the complete answer after validation at 119.14s.

The answer describes configuring specialized subagents and recommends them for multi-step work that clutters the main context or specialized domains needing custom instructions/tools. Manual spot-checking found the usage guidance consistent with selected quotations. The definition emphasizes configuration; this is not a formal semantic-quality evaluation. Existing citation checks establish valid references and exact source spans, not proof of entailment for every claim.

## Interpretation and cleanup

For this supported question on this laptop, with models warmed and an existing index, **one CPU+iGPU user waited about two minutes**. This is one run, not an average, percentile or guarantee for other questions. The earlier CPU-only 84.35s result is historical, with different generated content/token counts and system conditions; this is not a controlled CPU-versus-hybrid speed comparison. The five-user latencies of 6.1–10.7 minutes included substantial queueing and cannot be divided by five to predict a single-user latency.

The real full RAG run provides the relevant functional check; no application code changed, so the prior 28 offline regression tests were not rerun. Temporary models were unloaded and their server stopped. Main Ollama 11434 and Streamlit 8501 returned HTTP 200 afterwards. Raw results, model/tool traces, answers and logs remain ignored in `work/single-user-hybrid/hybrid/`; its harness is `work/single-user-hybrid/run.py`. No model files, index data or raw large outputs were committed.
