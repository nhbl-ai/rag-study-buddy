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
