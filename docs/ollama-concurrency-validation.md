# Five simultaneous users on CPU + Intel iGPU

Date: 2026-10-19. Branch: `feat-ollama`. RAG source revision: `5d231f7`.

## Requested scope

The user clarified that the intended test combines CPU and iGPU, rather than comparing separate CPU-only and iGPU-only modes. This report focuses on the combined configuration. A CPU-only burst finished before the clarification and remains archived in ignored `work/concurrency5/cpu/`; no standalone iGPU-only five-user run was performed.

Five prepared RagSession instances submitted five different questions simultaneously using a barrier. Each instance had its own vector store, empty history, model client, evidence files and citation anchors. The shared Ollama service hosted the model weights. This exercised the real RAG service and model/tool execution, rather than five browser sessions or frontend/network load.

## Hardware and settings

- Windows 11; Intel i7-13650HX, 16 GB RAM; Intel UHD / RaptorLake-S Mobile Graphics Controller.
- Ollama 0.40.5 on a temporary local-only server at localhost:11435; cloud features disabled.
- CUDA, ROCm and HIP device selection excluded the RTX 4050. Vulkan device 0 was verified as Intel iGPU in the startup log.
- Qwen3.5 4B Q4_K_M: benchmark-only `num_gpu=16`; runtime confirmed **16/33 layers offloaded to Intel**, with the remaining 17 on CPU. This is real partial model offloading, rather than merely CPU orchestration around a fully GPU-loaded model.
- EmbeddingGemma 300M: benchmark-only `num_gpu=0`, CPU inference.
- Context 8192; thinking disabled; output cap 1024; otherwise model-default sampling.
- One inference slot, matching the current application configuration, and at most two loaded models. Five simultaneous users therefore queue for shared inference; five parallel model slots were not enabled.
- Original 180-second client request/read timeout retained. It applies to individual model requests/stream reads, not the entire multi-call RAG workflow.
- Identical prepared snapshot of 321 passages from the existing public documentation corpus. Index construction, document downloads and cold model warmup were excluded from user latencies. Separate top-four retrieval probes were recorded for all five questions before dispatch.
- This is one layer split, not an optimization sweep or a claim that 16 layers is optimal.

## Measured results

All five sessions started within 0.006 seconds. Both model placement and the `-ngl 16` llama-server argument were recorded. Chat placement reported 3,998,639,714 bytes total and 1,886,010,735 bytes on the graphics device; iGPU graphics memory is shared system RAM. Embedding placement reported zero GPU memory.

| User | Question topic | Time from submission, seconds | Outcome | Citation links |
|---|---|---:|---|---:|
| 1 | Definition and when to use subagents | 478.46 | ReadTimeout waiting for synthesis; no answer accepted | 0 |
| 2 | Context bloat prevention | 442.41 | Passed workflow/citations | 4 |
| 3 | When to avoid subagents | 338.94 | Passed workflow/citations | 2 |
| 4 | CompiledSubAgent versus dictionary SubAgent | 376.16 | Cited partial answer + explicit insufficient information | 1 |
| 5 | Streaming and distinguishing subagent activity | 541.01 | Passed workflow/citations | 8 |

- Batch duration until all five clients terminated: **541.01s**, approximately 9 minutes 1 second.
- **4/5 sessions returned accepted cited responses**, including one partial response; **1/5 timed out**. This is one burst, not an 80% reliability estimate.
- All five searches, all **20 Deep Agents task executions**, all **20 analyst model calls** and all **20 nested read_file executions** completed successfully. Selected quotes were checked verbatim against each assigned passage; citation anchors stayed within their own user turn.
- The four accepted responses contained **15 valid citation links**. Structural validation does not prove semantic correctness or full question coverage. User 4 explicitly lacked enough selected evidence to complete the requested comparison.
- Mean latency among the four accepted responses: **424.63s**. Successful responses ranged from **338.94s to 541.01s**.
- User 1 submitted synthesis at about 298.46s after dispatch, received no first synthesis token and hit ReadTimeout 180.01s later, at 478.46s total. Its four subagent tasks had already passed. The application accepted no answer for this user.
- Lowest sampled free physical RAM: **0.46 GiB**. This includes other applications on the machine and is not a model-only memory allocation measurement.
- First internal model token latencies were about 13.93–70.98s. These include queuing and prefill and are not pure queue-wait measurements. Analyst tokens are private; the UI publishes the final answer after validation.

## Interpretation and limits

CPU+iGPU cooperation works technically and the required subagent/citation pipeline ran under simultaneous load. This particular configuration was slow and did not complete every client request successfully: five concurrent users experienced roughly 5.6–9 minutes before acceptance/failure, with one timeout. It is not sufficient evidence for a responsive or reliable five-user deployment.

The result applies to this Intel UHD, this model, this split and one inference slot. Different hardware, layer placement, output lengths and parallel settings require their own measurements. Default sampling produced different answer lengths; a single five-question burst is not a steady-state capacity or quality evaluation. No application source or main-server settings were changed for this experiment.

Raw per-user model timings, usage, task returns, retrieved passages, answers, errors, memory samples and device logs are in ignored `work/concurrency5/hybrid/`. The benchmark harness is `work/concurrency5/run.py`. The temporary server is unloaded and stopped after the measurement; the main Ollama server and Streamlit remain on their usual ports.
