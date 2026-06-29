# Evaluation

This directory adds the first measurable evaluation layer for the Ragent-inspired medical assistant refactor.

## Current Scope

The current scripts evaluate deterministic routing and FAQ retrieval:

- Expected intent
- Expected agent
- Expected tool
- Difficulty buckets
- FAQ Recall@K, MRR, nDCG@5, domain hit rate, priority hit rate, source hit rate

Run:

```bash
python evals/run_eval.py --task intent
python evals/run_eval.py --task tool
python evals/run_faq_retrieval_eval.py --mode keyword
python evals/dataset_report.py --dataset evals/datasets/intent_routing_eval.jsonl
python evals/dataset_report.py --dataset evals/datasets/tool_call_eval.jsonl
```

## Dataset Format

Each line is a JSON object:

```json
{
  "id": "intent-001",
  "query": "What are the latest COVID chest X-ray studies?",
  "expected_intent": "latest_medical_research",
  "expected_agent": "WEB_SEARCH_PROCESSOR_AGENT",
  "expected_tool": "web_search",
  "difficulty": "medium"
}
```

Image routing samples can include:

```json
{
  "has_image": true,
  "image_type": "CHEST X-RAY"
}
```

## Next Metrics

The next evaluation layers should add:

- Tool calls: parameter extraction accuracy and execution success rate
- Generation: RAGAS faithfulness and answer correctness
- Safety: unsafe block rate and false positive rate
- Engineering: latency, first-token time, token usage, and fallback rate

## FAQ Retrieval Evaluation

The FAQ retrieval dataset labels expected FAQ chunk IDs, domain, priority, and source organization:

```json
{
  "id": "faq-rag-001",
  "query": "头痛就是脑肿瘤吗？",
  "expected_faq_ids": ["FAQ-BT-001"],
  "expected_domain": "brain_tumor",
  "expected_priority": "P0",
  "expected_source_orgs": ["MedlinePlus"],
  "difficulty": "medium"
}
```

Metrics:

- `Recall@K`: whether any expected source appears in the top K chunks.
- `MRR`: how early the first relevant chunk appears.
- `nDCG`: whether more relevant chunks appear near the top.
- Channel hit rate: which retrieval channel contributed the final top chunks.
- Rerank delta: whether rerank improved the position of expected sources.

Current seed result on `evals/datasets/faq_retrieval_eval.jsonl`:

| Mode | Samples | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAQ keyword channel | 50 | 81.00% | 95.00% | 96.00% | 0.933 | 0.926 |
| Full RAG retrieval | 50 | 80.00% | 93.00% | 96.00% | 0.941 | 0.922 |

Full RAG mode:

```bash
DISABLE_CROSS_ENCODER_RERANKER=true python evals/run_faq_retrieval_eval.py --mode rag
```

Implemented metric helpers:

- `recall_at_k`
- `reciprocal_rank`
- `ndcg_at_k`
