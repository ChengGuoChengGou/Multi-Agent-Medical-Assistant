from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_agent.retrieval_channels import FAQKeywordSearchChannel, SearchContext  # noqa: E402
from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieve_with_keyword_channel(query: str, top_k: int) -> Dict[str, Any]:
    channel = FAQKeywordSearchChannel()
    result = channel.search(
        SearchContext(
            query=query,
            vectorstore=None,
            docstore=None,
            top_k=top_k,
        )
    )
    return {
        "chunks": result.chunks,
        "trace": {
            "channels": [result.to_trace()],
            "postprocessors": [],
            "final_chunk_count": len(result.chunks),
        },
    }


def retrieve_with_rag_engine(query: str, top_k: int) -> Dict[str, Any]:
    from config import Config
    from agents.rag_agent.medical_rag import MedicalRAG
    from agents.rag_agent.retrieval_channels import SearchContext

    config = Config()
    rag = MedicalRAG(config)
    vectorstore, docstore = rag.vector_store.load_vectorstore()
    chunks, picture_paths, trace = rag.retrieval_engine.retrieve(
        SearchContext(
            query=query,
            vectorstore=vectorstore,
            docstore=docstore,
            top_k=top_k,
        )
    )
    return {
        "chunks": chunks,
        "picture_paths": picture_paths,
        "trace": trace,
    }


def evaluate(rows: List[Dict[str, Any]], mode: str, top_k: int) -> Dict[str, Any]:
    retrieved_rows = []
    failures = []

    retriever = retrieve_with_keyword_channel if mode == "keyword" else retrieve_with_rag_engine

    for row in rows:
        result = retriever(row["query"], top_k=top_k)
        chunks = result["chunks"]
        retrieved_ids = [chunk.get("faq_id") or chunk.get("id", "") for chunk in chunks]
        expected_ids = row.get("expected_faq_ids", [])
        expected_domain = row.get("expected_domain")
        expected_priority = row.get("expected_priority")
        expected_source_orgs = set(row.get("expected_source_orgs", []))
        top_chunk = chunks[0] if chunks else {}

        evaluated = {
            **row,
            "retrieved_faq_ids": retrieved_ids[:top_k],
            "top_faq_id": top_chunk.get("faq_id"),
            "top_domain": top_chunk.get("domain"),
            "top_priority": top_chunk.get("priority"),
            "top_source_org": top_chunk.get("source_org"),
            "recall_at_1": recall_at_k(retrieved_ids, expected_ids, 1),
            "recall_at_3": recall_at_k(retrieved_ids, expected_ids, 3),
            "recall_at_5": recall_at_k(retrieved_ids, expected_ids, 5),
            "mrr": reciprocal_rank(retrieved_ids, expected_ids),
            "ndcg_at_5": ndcg_at_k(retrieved_ids, expected_ids, 5),
            "domain_hit": top_chunk.get("domain") == expected_domain,
            "priority_hit": top_chunk.get("priority") == expected_priority,
            "source_hit": top_chunk.get("source_org") in expected_source_orgs,
            "p0_top1": top_chunk.get("priority") == "P0",
            "trace": result["trace"],
        }
        retrieved_rows.append(evaluated)

        if evaluated["recall_at_5"] <= 0:
            failures.append(evaluated)

    return {
        "mode": mode,
        "top_k": top_k,
        "rows": retrieved_rows,
        "summary": summarize(retrieved_rows),
        "failures": failures,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}

    by_difficulty = defaultdict(list)
    by_domain = defaultdict(list)
    channel_counter = Counter()
    for row in rows:
        by_difficulty[row.get("difficulty", "unknown")].append(row)
        by_domain[row.get("expected_domain", "unknown")].append(row)
        for channel in row.get("trace", {}).get("channels", []):
            if channel.get("chunk_count", 0) > 0:
                channel_counter[channel.get("channel_name", "unknown")] += 1

    return {
        "samples": total,
        "recall_at_1": _mean(row["recall_at_1"] for row in rows),
        "recall_at_3": _mean(row["recall_at_3"] for row in rows),
        "recall_at_5": _mean(row["recall_at_5"] for row in rows),
        "mrr": _mean(row["mrr"] for row in rows),
        "ndcg_at_5": _mean(row["ndcg_at_5"] for row in rows),
        "domain_top1_accuracy": _mean(1.0 if row["domain_hit"] else 0.0 for row in rows),
        "priority_top1_accuracy": _mean(1.0 if row["priority_hit"] else 0.0 for row in rows),
        "source_top1_accuracy": _mean(1.0 if row["source_hit"] else 0.0 for row in rows),
        "p0_top1_rate": _mean(1.0 if row["p0_top1"] else 0.0 for row in rows),
        "by_difficulty": {
            key: {
                "samples": len(group),
                "recall_at_5": _mean(row["recall_at_5"] for row in group),
                "mrr": _mean(row["mrr"] for row in group),
            }
            for key, group in sorted(by_difficulty.items())
        },
        "by_domain": {
            key: {
                "samples": len(group),
                "recall_at_5": _mean(row["recall_at_5"] for row in group),
                "mrr": _mean(row["mrr"] for row in group),
            }
            for key, group in sorted(by_domain.items())
        },
        "channel_hit_count": dict(channel_counter),
    }


def print_summary(evaluation: Dict[str, Any], show_failures: bool) -> None:
    summary = evaluation["summary"]
    print("FAQ retrieval evaluation")
    print(f"- Mode: {evaluation['mode']}")
    print(f"- Samples: {summary['samples']}")
    print(f"- Recall@1: {summary['recall_at_1']:.2%}")
    print(f"- Recall@3: {summary['recall_at_3']:.2%}")
    print(f"- Recall@5: {summary['recall_at_5']:.2%}")
    print(f"- MRR: {summary['mrr']:.3f}")
    print(f"- nDCG@5: {summary['ndcg_at_5']:.3f}")
    print(f"- Top-1 domain accuracy: {summary['domain_top1_accuracy']:.2%}")
    print(f"- Top-1 priority accuracy: {summary['priority_top1_accuracy']:.2%}")
    print(f"- Top-1 source accuracy: {summary['source_top1_accuracy']:.2%}")
    print(f"- Top-1 P0 rate: {summary['p0_top1_rate']:.2%}")
    print(f"- Channel hits: {summary['channel_hit_count']}")

    print("\nBy difficulty")
    for difficulty, row in summary["by_difficulty"].items():
        print(
            f"- {difficulty}: samples={row['samples']} "
            f"Recall@5={row['recall_at_5']:.2%} MRR={row['mrr']:.3f}"
        )

    print("\nBy domain")
    for domain, row in summary["by_domain"].items():
        print(
            f"- {domain}: samples={row['samples']} "
            f"Recall@5={row['recall_at_5']:.2%} MRR={row['mrr']:.3f}"
        )

    if show_failures and evaluation["failures"]:
        print("\nFailures")
        for row in evaluation["failures"]:
            print(
                f"- {row['id']}: expected={row.get('expected_faq_ids')} "
                f"retrieved={row.get('retrieved_faq_ids')} query={row.get('query')!r}"
            )


def _mean(values: Any) -> float:
    value_list = list(values)
    return sum(value_list) / len(value_list) if value_list else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FAQ retrieval quality.")
    parser.add_argument(
        "--dataset",
        default="evals/datasets/faq_retrieval_eval.jsonl",
        help="Path to the FAQ retrieval JSONL dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "keyword"],
        default="rag",
        help="Use the full RAG retrieval engine or the FAQ keyword channel only.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", default=None, help="Optional path for detailed JSON output.")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(ROOT / args.dataset)
    evaluation = evaluate(rows, mode=args.mode, top_k=args.top_k)
    print_summary(evaluation, show_failures=args.show_failures)

    if args.json_output:
        output_path = ROOT / args.json_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if not evaluation["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
