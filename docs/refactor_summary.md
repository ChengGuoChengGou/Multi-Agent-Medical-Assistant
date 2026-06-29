# Ragent-Inspired Refactor Summary

This document summarizes the main engineering changes made to turn the original Multi-Agent Medical Assistant into an interview-ready Agentic RAG project.

## 1. Routing And Intent Layer

Added:

- `agents/router/medical_intents.py`
- `agents/router/intent_router.py`
- `agents/tracing/agent_trace.py`

Integrated into:

- `agents/agent_decision.py`

What changed:

- Added 15 medical/system/safety intents.
- Added deterministic router before the existing LLM router.
- Added route metadata: intent, tool, confidence, reason, matched keywords.
- Added trace records for thought/action/observation style routing explanation.

## 2. Agentic RAG Retrieval

Added:

- `agents/rag_agent/retrieval_channels.py`
- `agents/rag_agent/postprocessors.py`
- `agents/rag_agent/react_controller.py`

Integrated into:

- `agents/rag_agent/medical_rag.py`
- `agents/rag_agent/vectorstore_qdrant.py`

What changed:

- Split retrieval into intent-directed and global Qdrant channels.
- Added postprocessor chain: deduplication, rerank, confidence calculation.
- Added bounded ReAct retrieval self-correction.
- Added `react_trace` and `retrieval_trace` to RAG responses.
- Added configurable `react_max_steps` and `react_timeout_seconds`.

## 3. Ingestion Observability

Added:

- `agents/rag_agent/ingestion_pipeline.py`
- `docs/ingestion_pipeline.md`

Integrated into:

- `agents/rag_agent/medical_rag.py`

What changed:

- Split document ingestion into parse, image summary, format, chunk, index nodes.
- Added per-node status, duration, error, and output summary.
- Added `ingestion_trace` for single-file ingestion and `file_results` for directory ingestion.

## 4. Model Reliability And Cost Control

Added:

- `core/model_gateway.py`
- `core/circuit_breaker.py`
- `core/retry_policy.py`
- `core/token_tracker.py`
- `core/rate_limiter.py`

Integrated into:

- `agents/rag_agent/query_expander.py`
- `agents/rag_agent/response_generator.py`

What changed:

- Added centralized model invocation gateway.
- Added retry policy and three-state circuit breaker.
- Added FIFO rate limiter with queue wait metadata.
- Added heuristic token usage estimates.

## 5. Prompt Governance

Added:

- `core/prompt_registry.py`
- `prompts/manifest.json`
- `prompts/rag/query_expansion.v1.txt`
- `prompts/rag/response_generation.v1.txt`

Integrated into:

- `agents/rag_agent/query_expander.py`
- `agents/rag_agent/response_generator.py`

What changed:

- Extracted key prompts out of Python strings.
- Added active-version control through `manifest.json`.
- Added prompt metadata to model gateway calls for later evaluation.

## 6. Evaluation And Demo

Added:

- `evals/run_eval.py`
- `evals/metrics.py`
- `evals/dataset_report.py`
- `evals/datasets/intent_routing_eval.jsonl`
- `evals/datasets/tool_call_eval.jsonl`
- `demo/offline_agentic_walkthrough.py`
- `scripts/run_local_checks.py`

What changed:

- Expanded intent routing eval to 50 samples.
- Expanded tool-call eval to 25 samples.
- Added difficulty and distribution reporting.
- Added local non-network demo and unified local check runner.

Current seed baseline:

- Intent accuracy: 50/50 = 100%
- Agent accuracy: 50/50 = 100%
- Tool accuracy: 50/50 = 100%
- Tool-call eval: 25/25 = 100%

## 7. Package Structure Cleanup

Added:

- `agents/rag_agent/medical_rag.py`
- `docs/engineering_cleanup.md`

What changed:

- Moved heavy `MedicalRAG` implementation out of `agents/rag_agent/__init__.py`.
- Kept `from agents.rag_agent import MedicalRAG` backward-compatible through a lazy proxy.
- Made lightweight modules importable without Docling/Qdrant/model dependencies.

## Suggested Review Order

1. `README.md`
2. `docs/refactor_summary.md`
3. `agents/router/`
4. `agents/rag_agent/react_controller.py`
5. `agents/rag_agent/retrieval_channels.py`
6. `core/model_gateway.py`
7. `evals/run_eval.py`
8. `demo/offline_agentic_walkthrough.py`
