import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


FAQ_HEADING_PATTERN = re.compile(r"^##\s+(FAQ-[A-Z]+-\d+)\s*$", re.MULTILINE)


@dataclass
class FAQEntry:
    faq_id: str
    source_file: str
    fields: Dict[str, str]

    @property
    def question(self) -> str:
        return self.fields.get("question", "")

    @property
    def answer(self) -> str:
        return self.fields.get("answer", "")

    def to_chunk(self) -> str:
        source_org = self.fields.get("source_org", "")
        source_url = self.fields.get("source_url", "")
        lines = [
            f"FAQ ID: {self.faq_id}",
            f"Domain: {self.fields.get('domain', '')}",
            f"Priority: {self.fields.get('priority', '')}",
            f"Intent: {self.fields.get('intent', '')}",
            f"Risk Level: {self.fields.get('risk_level', '')}",
            f"Need Doctor: {self.fields.get('need_doctor', '')}",
            f"Question: {self.question}",
            f"Answer: {self.answer}",
            f"Keywords: {self.fields.get('keywords', '')}",
            f"Source: {source_org} {source_url}".strip(),
        ]
        return "\n".join(line for line in lines if line.strip())

    def to_metadata(self) -> Dict[str, str]:
        metadata = {
            "content_type": "patient_faq",
            "faq_id": self.faq_id,
            "domain": self.fields.get("domain", ""),
            "priority": self.fields.get("priority", ""),
            "intent": self.fields.get("intent", ""),
            "risk_level": self.fields.get("risk_level", ""),
            "need_doctor": self.fields.get("need_doctor", ""),
            "source_org": self.fields.get("source_org", ""),
            "source_url": self.fields.get("source_url", ""),
            "source_type": self.fields.get("source_type", ""),
            "keywords": self.fields.get("keywords", ""),
            "source_file": self.source_file,
        }
        return {key: value for key, value in metadata.items() if value != ""}


def load_faq_file(path: str) -> List[FAQEntry]:
    faq_path = Path(path)
    text = faq_path.read_text(encoding="utf-8")
    domain = _read_header_value(text, "Domain")
    entries: List[FAQEntry] = []

    matches = list(FAQ_HEADING_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        faq_id = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        fields = _parse_fields(block)
        fields.setdefault("domain", domain)
        if not fields.get("question") or not fields.get("answer"):
            raise ValueError(f"FAQ entry {faq_id} is missing question or answer")
        entries.append(
            FAQEntry(
                faq_id=faq_id,
                source_file=str(faq_path),
                fields=fields,
            )
        )

    if not entries:
        raise ValueError(f"No FAQ entries found in {path}")
    return entries


def load_faq_directory(directory: str) -> List[FAQEntry]:
    faq_dir = Path(directory)
    if not faq_dir.is_dir():
        raise ValueError(f"FAQ directory not found: {directory}")

    entries: List[FAQEntry] = []
    for path in sorted(faq_dir.glob("*_faq.md")):
        entries.extend(load_faq_file(str(path)))
    if not entries:
        raise ValueError(f"No FAQ files found in {directory}")
    return entries


def _parse_fields(block: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def _read_header_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""
