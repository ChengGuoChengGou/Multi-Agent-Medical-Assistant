# Ragent-Inspired Refactor Plan

## Why Reference Ragent

Ragent is valuable as an enterprise RAG reference because it separates business logic from reusable AI infrastructure. The medical assistant can borrow the architecture ideas without changing its Python/FastAPI stack:

- Intent tree instead of a single router prompt
- Query rewrite and sub-question split before retrieval
- Multi-channel retrieval with post-processors
- Prompt templates managed by scene
- Ingestion pipeline with per-node status
- Model gateway with failover and circuit breaker
- Trace records for each RAG and Agent step
- Evaluation datasets and measurable optimization

## Target Architecture

```text
FastAPI
  -> Agentic medical pipeline
     -> load conversation memory
     -> rewrite and split query
     -> resolve medical intents
     -> guidance or clarification
     -> run retrieval/tools/CV agents
     -> post-process evidence
     -> build prompt by scene
     -> call model gateway
     -> guardrails and human validation
     -> trace and metrics
```

## Proposed Python Module Layout

```text
agents/
  router/
    medical_intents.py
    intent_router.py
  tracing/
    agent_trace.py
  tools/
    tool_registry.py
    rag_tool.py
    web_search_tool.py
    image_tool.py
  rag_agent/
    retrieval_channels.py
    postprocessors.py
    prompt_templates.py

core/
  model_gateway.py
  circuit_breaker.py
  retry_policy.py
  rate_limiter.py
  token_tracker.py

evals/
  datasets/
  run_eval.py
  metrics.py
```

## Phase 1: Documentation and Evaluation Baseline

Deliverables:

- Business scenario document
- Medical intent taxonomy
- Evaluation JSONL datasets
- Evaluation script that can score routing results

Interview questions covered:

- What business scenario does the system serve?
- How many user intents did you design?
- How is the knowledge base organized?
- How do you evaluate routing and tool calls?

## Phase 2: Explainable Router and Trace

Deliverables:

- `MedicalIntent` definitions
- `IntentRouter` output schema
- `AgentTrace` with thought/action/observation events
- Optional integration into `agent_decision.py`

Trace event format:

```json
{
  "step": 1,
  "type": "thought",
  "name": "route-intent",
  "content": "Question mentions recent research, web search is needed.",
  "metadata": {
    "confidence": 0.91
  }
}
```

Interview questions covered:

- How does Agent decide whether to retrieve, call tools, or ask back?
- How can you explain one complex ReAct-style process?
- How do you prevent uncontrolled loops?

## Phase 3: Multi-Channel Retrieval

Deliverables:

- `SearchChannel` interface
- Qdrant hybrid channel
- Intent-directed channel
- Global fallback channel
- Web search channel for recent medical info
- Post-processor chain: deduplication, rerank, source filter, confidence calculator

Interview questions covered:

- What happens when retrieval quality is insufficient?
- How is Agentic RAG different from naive RAG?
- How are KB and tool results separated in prompts?

## Phase 4: Model Gateway and Reliability

Deliverables:

- Unified LLM gateway for chat, embedding, and rerank calls
- Three-state circuit breaker: `CLOSED`, `OPEN`, `HALF_OPEN`
- Retry policy by error class
- Streaming first-token probe
- Queue/semaphore based concurrency control
- Token and latency tracking

Interview questions covered:

- What if the model API is down?
- How do you judge success in streaming mode?
- How do you handle model concurrency limits?
- How do you control token cost?

## Phase 5: Ingestion Pipeline

Deliverables:

- Document task state model
- Node-level logs for parse, image summary, chunk, embed, index
- Retry failed node or failed document
- Metadata for document source, disease domain, version, and chunk count

Interview questions covered:

- What happens from document upload to searchable knowledge?
- What if one step fails?
- How many documents are in the knowledge base and how are they organized?

## Phase 6: Full Evaluation Loop

Deliverables:

- 100+ labeled samples
- Baseline report
- Optimized report
- Metrics for intent, tool call, retrieval, generation, safety

Target metrics:

- Intent accuracy
- Agent routing accuracy
- Tool-call accuracy
- Recall@K
- MRR / nDCG
- RAGAS faithfulness
- Answer correctness
- Safety block rate
- Average latency and token usage

## Scope Control

The first implementation should not replace the whole LangGraph workflow. Add the new router and trace in parallel, then gradually route traffic through it. This keeps the project runnable while making the architecture more defensible.
