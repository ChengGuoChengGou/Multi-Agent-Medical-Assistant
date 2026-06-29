# Multi-Channel Retrieval

This refactor adapts the multi-channel retrieval idea from Ragent to the medical assistant while keeping the existing Qdrant and CrossEncoder implementation.

## Why Split Retrieval Into Channels

The previous RAG flow was:

```text
query expansion -> Qdrant retrieval -> CrossEncoder rerank -> answer
```

That works, but it is hard to explain or extend. The new flow is:

```text
query expansion
  -> enabled SearchChannels
  -> merge chunks
  -> PostProcessor chain
  -> response generation
```

This makes each retrieval decision measurable:

- Which channels were enabled
- How many chunks each channel returned
- How long each channel took
- How many chunks each postprocessor removed or retained
- How many reference images were found

## Current Channels

| Channel | Type | Purpose |
| --- | --- | --- |
| `faq_keyword_search` | `FAQ_KEYWORD` | Uses curated FAQ questions, keywords, IDF-style lexical scoring, and abbreviation boosts for patient-style FAQ recall. |
| `intent_directed_medical_search` | `INTENT_DIRECTED` | Detects medical domain terms such as brain tumor, chest X-ray, skin lesion, and diabetes, then filters and boosts matching chunks. |
| `qdrant_global_search` | `VECTOR_HYBRID_GLOBAL` | Uses the existing Qdrant hybrid retrieval as a broad recall fallback. |

The current repository stores all chunks in one Qdrant collection, so intent-directed search is implemented as domain filtering and score boosting on top of Qdrant results. If the knowledge base is later split by domain or metadata, this channel can become a real collection/metadata-filtered retriever without changing the RAG pipeline.

## Current PostProcessors

| Processor | Order | Purpose |
| --- | --- | --- |
| `deduplication` | 1 | Merges duplicate chunks from multiple channels and keeps the highest-scored copy. |
| `faq_priority_boost` | 5 | Boosts patient FAQ chunks by P0/P1/P2 priority, risk level, and doctor-needed flag before reranking. |
| `cross_encoder_rerank` | 10 | Reuses the existing HuggingFace CrossEncoder reranker and extracts reference image paths. |
| `confidence_calculator` | 20 | Adds a retrieval confidence field using score, rank, and channel hints. |

## Trace Shape

`MedicalRAG.process_query()` now returns `retrieval_trace` in the response:

```json
{
  "channels": [
    {
      "channel_name": "intent_directed_medical_search",
      "channel_type": "INTENT_DIRECTED",
      "chunk_count": 3,
      "latency_ms": 120,
      "confidence": 0.78,
      "error": null
    }
  ],
  "postprocessors": [
    {
      "name": "deduplication",
      "before_count": 8,
      "after_count": 5
    }
  ],
  "initial_chunk_count": 8,
  "final_chunk_count": 3,
  "picture_reference_count": 2
}
```

This directly supports interview questions about retrieval correction, observability, and engineering tradeoffs.

## FAQ Retrieval Evaluation

The project includes `evals/datasets/faq_retrieval_eval.jsonl` with 50 labeled FAQ retrieval cases across seven domains. Current local seed results:

| Mode | Samples | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAQ keyword channel | 50 | 81.00% | 95.00% | 96.00% | 0.933 | 0.926 |
| Full RAG retrieval | 50 | 80.00% | 93.00% | 96.00% | 0.941 | 0.922 |

Run:

```bash
python evals/run_faq_retrieval_eval.py --mode keyword
DISABLE_CROSS_ENCODER_RERANKER=true python evals/run_faq_retrieval_eval.py --mode rag
```

## Failure Behavior

- A failed channel returns an empty result with an error field.
- Other channels continue running.
- A failed postprocessor is not yet wrapped; this should be hardened in the next iteration.
- If no chunks remain, the existing response generator produces a low-confidence response and the agent graph can escalate to web search.

## Next Extensions

1. Add metadata-aware Qdrant filters for FAQ domain and priority fields.
2. Add a keyword channel for exact disease names, paper titles, and abbreviations.
3. Add source quality filtering for medical references.
4. Add retrieval evaluation with Recall@K, MRR, and nDCG.
5. Add per-channel latency and hit-rate reporting in the admin/debug view.
