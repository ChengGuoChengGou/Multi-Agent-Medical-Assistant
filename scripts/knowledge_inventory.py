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


def infer_domain(name: str) -> str:
    normalized = name.lower()
    if "brain" in normalized or "tumor" in normalized:
        return "brain_tumor"
    if "covid" in normalized or "chest" in normalized or "xray" in normalized:
        return "chest_xray_covid"
    if "skin" in normalized or "lesion" in normalized:
        return "skin_lesion"
    if "diabetes" in normalized:
        return "diabetes"
    if "cardio" in normalized or "heart" in normalized or "stroke" in normalized:
        return "cardiovascular"
    if "medication" in normalized or "medicine" in normalized or "drug" in normalized:
        return "medication_safety"
    return "general_medical"


def collect_pdf_documents(directory: Path, status: str) -> List[Dict[str, Any]]:
    documents = []
    for path in sorted(directory.glob("*.pdf")) if directory.exists() else []:
        documents.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)),
                "domain": infer_domain(path.name),
                "type": "pdf",
                "status": status,
                "size_bytes": path.stat().st_size,
            }
        )
    return documents


def collect_faq_stats(faq_dir: Path) -> Dict[str, Any]:
    from agents.rag_agent.faq_loader import load_faq_directory

    if not faq_dir.exists():
        return {
            "documents": [],
            "entries": 0,
            "domain_counts": {},
            "priority_counts": {},
            "source_org_counts": {},
        }

    entries = load_faq_directory(str(faq_dir))
    documents = []
    entries_by_file = Counter(Path(entry.source_file).name for entry in entries)
    for path in sorted(faq_dir.glob("*_faq.md")):
        documents.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)),
                "domain": infer_domain(path.name),
                "type": "patient_faq_markdown",
                "status": "runtime_keyword_available",
                "entries": entries_by_file[path.name],
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "documents": documents,
        "entries": len(entries),
        "domain_counts": dict(Counter(entry.fields.get("domain", "unknown") for entry in entries)),
        "priority_counts": dict(Counter(entry.fields.get("priority", "unknown") for entry in entries)),
        "source_org_counts": dict(Counter(entry.fields.get("source_org", "unknown") for entry in entries)),
    }


def collect_parsed_doc_stats(parsed_dir: Path) -> Dict[str, Any]:
    grouped: Dict[str, Counter[str]] = defaultdict(Counter)
    if not parsed_dir.exists():
        return {"total_files": 0, "by_document": {}}

    for path in parsed_dir.glob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if "-picture-" in stem:
            base, kind = stem.split("-picture-", 1)[0], "pictures"
        elif "-table-" in stem:
            base, kind = stem.split("-table-", 1)[0], "tables"
        elif stem.rsplit("-", 1)[-1].isdigit():
            base, kind = stem.rsplit("-", 1)[0], "page_images"
        else:
            base, kind = stem, "other"
        grouped[base][kind] += 1

    return {
        "total_files": sum(sum(counter.values()) for counter in grouped.values()),
        "by_document": {name: dict(counter) for name, counter in sorted(grouped.items())},
    }


def qdrant_count(qdrant_dir: Path, collection_name: str) -> int | None:
    if not qdrant_dir.exists():
        return None
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(qdrant_dir))
        count = client.count(collection_name=collection_name, exact=True).count
        client.close()
        return count
    except Exception:
        return None


def build_inventory(collection_name: str) -> Dict[str, Any]:
    raw_documents = collect_pdf_documents(ROOT / "data" / "raw", "indexed")
    candidate_documents = collect_pdf_documents(ROOT / "data" / "raw_extras", "candidate_not_indexed")
    faq_stats = collect_faq_stats(ROOT / "data" / "faq")
    parsed_stats = collect_parsed_doc_stats(ROOT / "data" / "parsed_docs")
    qdrant_points = qdrant_count(ROOT / "data" / "qdrant_db", collection_name)
    docstore_files = len(list((ROOT / "data" / "docs_db").glob("*"))) if (ROOT / "data" / "docs_db").exists() else 0

    domain_counts = Counter()
    for document in raw_documents + candidate_documents + faq_stats["documents"]:
        domain_counts[document["domain"]] += document.get("entries", 1)

    return {
        "collection_name": collection_name,
        "indexed_pdf_documents": len(raw_documents),
        "candidate_pdf_documents": len(candidate_documents),
        "faq_documents": len(faq_stats["documents"]),
        "faq_entries": faq_stats["entries"],
        "qdrant_points": qdrant_points,
        "docstore_files": docstore_files,
        "raw_documents": raw_documents,
        "candidate_documents": candidate_documents,
        "faq": faq_stats,
        "parsed_docs": parsed_stats,
        "domain_counts_including_candidates_and_faq_entries": dict(domain_counts),
    }


def print_summary(inventory: Dict[str, Any]) -> None:
    print("Knowledge inventory")
    print(f"- Collection: {inventory['collection_name']}")
    print(f"- Indexed PDFs: {inventory['indexed_pdf_documents']}")
    print(f"- Candidate PDFs not indexed: {inventory['candidate_pdf_documents']}")
    print(f"- FAQ markdown files: {inventory['faq_documents']}")
    print(f"- FAQ entries: {inventory['faq_entries']}")
    print(f"- Qdrant points: {inventory['qdrant_points']}")
    print(f"- Docstore files: {inventory['docstore_files']}")
    print(f"- Parsed doc files: {inventory['parsed_docs']['total_files']}")

    print("\nFAQ priority distribution")
    for priority, count in sorted(inventory["faq"]["priority_counts"].items()):
        print(f"- {priority}: {count}")

    print("\nFAQ domain distribution")
    for domain, count in sorted(inventory["faq"]["domain_counts"].items()):
        print(f"- {domain}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize local medical knowledge base inventory.")
    parser.add_argument("--collection", default="medical_assistance_rag")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    inventory = build_inventory(args.collection)
    print_summary(inventory)

    if args.json_output:
        output_path = ROOT / args.json_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
