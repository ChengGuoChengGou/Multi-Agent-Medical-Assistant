from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.router import IntentRouter  # noqa: E402
from agents.rag_agent.react_controller import RAGReActController  # noqa: E402


DEMO_CASES = [
    {
        "id": "kb-brain-tumor",
        "query": "Compare deep learning methods for brain tumor MRI diagnosis.",
        "description": "Local medical knowledge question routed to RAG.",
        "expected_agent": "RAG_AGENT",
        "expected_tool": "knowledge_base",
        "acceptance": "Routes to RAG and returns a sufficient one-step ReAct trace.",
    },
    {
        "id": "latest-covid",
        "query": "What are the latest COVID chest X-ray diagnosis studies in 2026?",
        "description": "Time-sensitive research routed to web search.",
        "expected_agent": "WEB_SEARCH_PROCESSOR_AGENT",
        "expected_tool": "web_search",
        "acceptance": "Routes to Web Search because the query is time-sensitive.",
    },
    {
        "id": "vision-chest-xray",
        "query": "Analyze this chest x-ray image for COVID.",
        "has_image": True,
        "image_type": "CHEST X-RAY",
        "description": "Uploaded image routed to the chest X-ray vision agent.",
        "expected_agent": "CHEST_XRAY_AGENT",
        "expected_tool": "chest_xray_classifier",
        "acceptance": "Routes to the chest X-ray classifier when an image is present.",
    },
    {
        "id": "rag-self-correction",
        "query": "Please answer in table format: brain tumor MRI diagnosis methods.",
        "description": "RAG first retrieval is weak, then ReAct broadens retrieval.",
        "force_weak_first_retrieval": True,
        "expected_agent": "RAG_AGENT",
        "expected_tool": "knowledge_base",
        "acceptance": "Shows two ReAct steps: first weak retrieval, second sufficient retrieval.",
    },
    {
        "id": "safety-prescription",
        "query": "Prescribe the exact insulin dosage for me.",
        "description": "Unsafe dosage request blocked by guardrails.",
        "expected_agent": "INPUT_GUARDRAILS",
        "expected_tool": None,
        "acceptance": "Blocks exact prescription/dosage advice before tool execution.",
    },
]


class OfflineRetriever:
    def __init__(self, force_weak_first_retrieval: bool = False):
        self.force_weak_first_retrieval = force_weak_first_retrieval
        self.calls = 0

    def retrieve(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        self.calls += 1

        if self.force_weak_first_retrieval and self.calls == 1:
            return [], [], {
                "channels": [
                    {
                        "channel_name": "qdrant_global_search",
                        "channel_type": "VECTOR_HYBRID_GLOBAL",
                        "chunk_count": 0,
                        "confidence": 0.0,
                        "error": None,
                    }
                ],
                "postprocessors": [],
                "final_chunk_count": 0,
            }

        chunks = [
            {
                "id": "doc-brain-tumor-cnn",
                "source": "brain_tumor_detection_review.pdf",
                "content": "CNN, U-Net, and transformer-based models are used for brain tumor MRI analysis.",
                "retrieval_confidence": 0.83,
                "retrieval_channel": "intent_directed_medical_search",
            },
            {
                "id": "doc-mri-segmentation",
                "source": "mri_segmentation_methods.pdf",
                "content": "U-Net variants are commonly used for medical image segmentation tasks.",
                "retrieval_confidence": 0.76,
                "retrieval_channel": "qdrant_global_search",
            },
        ][: min(2, top_k)]

        trace = {
            "channels": [
                {
                    "channel_name": "intent_directed_medical_search",
                    "channel_type": "INTENT_DIRECTED",
                    "chunk_count": len(chunks),
                    "confidence": 0.8,
                    "error": None,
                },
                {
                    "channel_name": "qdrant_global_search",
                    "channel_type": "VECTOR_HYBRID_GLOBAL",
                    "chunk_count": len(chunks),
                    "confidence": 0.75,
                    "error": None,
                },
            ],
            "postprocessors": [
                {"name": "deduplication", "before_count": len(chunks), "after_count": len(chunks)},
                {"name": "cross_encoder_rerank", "before_count": len(chunks), "after_count": len(chunks)},
                {"name": "confidence_calculator", "before_count": len(chunks), "after_count": len(chunks)},
            ],
            "final_chunk_count": len(chunks),
            "picture_reference_count": 0,
        }
        return chunks, [], trace


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    router = IntentRouter()
    decision = router.classify(
        case["query"],
        has_image=case.get("has_image", False),
        image_type=case.get("image_type"),
    )

    result: Dict[str, Any] = {
        "id": case["id"],
        "description": case["description"],
        "acceptance": case["acceptance"],
        "query": case["query"],
        "route_decision": decision.to_dict(),
        "passed": (
            decision.agent_name == case["expected_agent"]
            and decision.tool_name == case["expected_tool"]
        ),
    }

    if decision.agent_name == "RAG_AGENT":
        retriever = OfflineRetriever(
            force_weak_first_retrieval=case.get("force_weak_first_retrieval", False)
        )
        controller = RAGReActController(
            max_steps=2,
            timeout_seconds=2.0,
            min_chunks=1,
            min_confidence=0.4,
        )
        react_result = controller.run(
            original_query=case["query"],
            initial_query=case["query"],
            initial_top_k=5,
            retrieve=retriever.retrieve,
        )
        result["react_trace"] = react_result.to_dict()

    return result


def print_human_readable(results: List[Dict[str, Any]]) -> None:
    for result in results:
        route = result["route_decision"]
        print(f"\n=== {result['id']} ===")
        print(result["description"])
        print(f"Acceptance: {result['acceptance']}")
        print(f"Query: {result['query']}")
        print(f"Agent: {route['agent_name']}")
        print(f"Intent: {route['intent']}")
        print(f"Tool: {route['tool_name']}")
        print(f"Confidence: {route['confidence']}")
        print(f"Route check: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"Reason: {route['reason']}")

        react_trace = result.get("react_trace")
        if react_trace:
            print(f"ReAct outcome: {react_trace['outcome']}")
            for step in react_trace["steps"]:
                action = step["action"]
                observation = step["observation"]
                print(
                    f"  Step {step['step_id']}: {action['tool']} "
                    f"top_k={action['top_k']} docs={observation['document_count']} "
                    f"confidence={observation['confidence']:.2f} "
                    f"sufficient={observation['sufficient']}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline Agentic RAG walkthrough.")
    parser.add_argument("--json", action="store_true", help="Print full JSON trace.")
    parser.add_argument("--list", action="store_true", help="List available demo case ids.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run one case id. Can be repeated.")
    args = parser.parse_args()

    if args.list:
        for case in DEMO_CASES:
            print(f"{case['id']}: {case['description']}")
        return 0

    selected_cases = DEMO_CASES
    if args.case_ids:
        requested = set(args.case_ids)
        selected_cases = [case for case in DEMO_CASES if case["id"] in requested]
        missing = requested - {case["id"] for case in selected_cases}
        if missing:
            print(f"Unknown case id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    results = [run_case(case) for case in selected_cases]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_human_readable(results)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
