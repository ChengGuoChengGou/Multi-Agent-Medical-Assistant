# Interview Question Coverage

This document maps the planned Ragent-inspired refactor to the interview questions this project should be able to answer.

## Business Scenario

| Question | Project Answer |
| --- | --- |
| What business scenario does the system serve? | Medical education, research assistance, and clinical decision support exploration. See `docs/business_scenario.md`. |
| Who are the target users? | Medical learners, healthcare assistants, clinicians or reviewers, and general users with medical questions. |
| How many user intents did you design? | 15 leaf intents across system, medical QA, medical image, and safety domains. See `docs/intent_taxonomy.md`. |
| Give 3 typical intents. | Literature-grounded QA routes to RAG; latest medical research routes to web search; chest X-ray upload routes to the chest X-ray agent. |
| What if the system cannot answer? | Guardrails block unsafe input, RAG escalates to web search on low confidence, and final responses state insufficient information instead of hallucinating. |
| How is the knowledge base designed? | Documents are grouped by medical task domain: brain tumor, chest X-ray/COVID, skin lesion, and optional general references. |

## Agentic RAG

| Question | Project Answer |
| --- | --- |
| What is ReAct? | A Reasoning + Acting loop: thought, tool action, observation, then final answer. The current project uses controlled LangGraph routing and is being refactored toward traceable Agentic RAG. |
| How is it different from a fixed pipeline? | A fixed pipeline always retrieves and generates. Agentic RAG decides whether to use KB, web search, image tools, clarification, or refusal. |
| How does Agent choose KB/tool/clarification? | `agents/router/IntentRouter` returns a structured `RouteDecision` with intent, agent, tool, confidence, and clarification flag. |
| How does it self-correct poor retrieval? | Planned multi-channel retrieval uses intent-directed search first, global retrieval as fallback, rerank, and web search escalation for low confidence. |
| How to prevent ReAct dead loops? | Keep a controlled loop with max steps, repeated tool-call detection, timeout, and explicit final fallback. |
| How are KB and tool results separated? | Prompt scenes should separate `kb_context`, `tool_context`, and mixed evidence, following the Ragent-style prompt template design. |

## Evaluation

| Question | Project Answer |
| --- | --- |
| How large is the evaluation set? | Initial baseline contains 12 intent-routing samples and 5 tool-call samples. It should be expanded to 100+ before claiming production-quality metrics. |
| What layers are evaluated? | Current layer: intent, agent, and tool routing. Planned layers: retrieval, tool parameters, generation, safety, latency, and token cost. |
| How is tool-call accuracy evaluated? | Each sample has `expected_tool` and `expected_agent`; `evals/run_eval.py` compares actual router output against labels. |
| What optimization was made from evaluation? | Initial eval exposed a false route from "brain tumors" to the brain MRI agent. The router now requires explicit vision-action wording for text-only vision routing. |
| What are RAGAS faithfulness and answer correctness? | Faithfulness checks whether the answer is supported by retrieved context. Answer correctness checks whether the final answer matches the reference answer. |

## Engineering Ability

| Question | Project Answer |
| --- | --- |
| What if the model API fails? | `core/model_gateway.py` adds retry and circuit-breaker protection for model calls. Provider-level fallback is the next extension. |
| What is the circuit breaker state transition? | `CLOSED`: normal calls; `OPEN`: block after failures; `HALF_OPEN`: allow one probe after cooldown and recover or reopen. |
| How to judge streaming success? | `await_first_packet()` defines first-token probing: the stream is healthy only after the first valid chunk arrives before timeout. |
| How to handle concurrency limits? | Planned queue/semaphore based limiter with atomic acquisition and timeout fallback. |
| What happens from upload to searchable document? | Parse, image summary, markdown formatting, semantic chunking, embedding, vector indexing, and docstore persistence. |
| How are Prompt templates managed? | Planned file-based prompt templates by scene: router, RAG, web search, image validation, safety, and mixed KB/tool answers. |
| How to control token cost? | Limit chat history, top-k, context length, rerank top-k, cache stable routing, and track per-stage token usage via the model gateway. |

## Comparative Advantage

| Question | Project Answer |
| --- | --- |
| Difference from LangChain/Dify quick setup? | This project exposes routing, trace, retrieval channels, validation, and model reliability as code-level extension points instead of black-box configuration. |
| Difference from naive RAG? | Naive RAG always retrieves from KB. This project chooses among KB, web search, image models, safety refusal, and clarification. |
| Hardest technical challenge? | Reliable orchestration across medical text, medical literature, web evidence, CV model output, guardrails, and human validation. |

## Current Baseline Command

```bash
python evals/run_eval.py
```

Current local result on the seed dataset:

```text
Intent accuracy: 12/12 = 100.00%
Agent accuracy: 12/12 = 100.00%
Tool accuracy: 12/12 = 100.00%
```

This is a seed set result, not a production metric. The next target is a 100+ sample evaluation set with optimized-before/after reports.
