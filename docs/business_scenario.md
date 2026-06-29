# Business Scenario

## Project Positioning

Multi-Agent Medical Assistant is a medical AI assistant for education, research assistance, and clinical decision support exploration. It does not replace licensed clinicians or issue final diagnoses. The system is designed to help users retrieve medical literature, understand medical concepts, analyze supported medical images, and escalate uncertain or high-risk outputs to safer fallback paths.

## Target Users

| User | Primary Need | System Boundary |
| --- | --- | --- |
| Medical learners | Learn disease concepts, imaging findings, and research context | Provide literature-grounded explanations |
| Healthcare assistants | Quickly search internal medical materials and summarize evidence | Keep answers traceable to sources |
| Clinicians or reviewers | Review AI-assisted image analysis results | Require human validation for vision outputs |
| General users | Understand medical topics in plain language | Add disclaimers and avoid prescriptions or final diagnosis |

## Core Scenarios

1. Medical knowledge Q&A
   - Stable knowledge questions such as disease definitions, symptoms, diagnosis methods, and treatment background.
   - Routed to conversation or RAG depending on whether local medical literature is needed.

2. Literature-grounded medical RAG
   - Questions that require evidence from ingested medical papers or reports.
   - Uses document parsing, semantic chunking, hybrid retrieval, reranking, and source display.

3. Latest medical research lookup
   - Time-sensitive questions such as recent studies, outbreaks, or updated research findings.
   - Routed to web search instead of relying only on the local knowledge base.

4. Medical image assistance
   - Uploaded images are classified as chest X-ray, skin lesion, brain MRI, or unsupported.
   - Supported tasks call the corresponding computer vision model and require human validation.

5. Safety and uncertainty handling
   - Unsafe input is blocked by guardrails.
   - Low-confidence RAG results are escalated to web search or answered with an explicit limitation.

## Typical User Intents

| Intent | Example | Expected Route |
| --- | --- | --- |
| General chat | "Hello, what can you do?" | Conversation agent |
| Medical concept | "What are common types of brain tumors?" | RAG agent |
| Literature explanation | "Summarize deep learning methods for brain tumor detection." | RAG agent |
| Latest research | "What are recent COVID chest X-ray diagnosis studies?" | Web search processor |
| Chest X-ray image | User uploads a chest X-ray image | Chest X-ray agent |
| Skin lesion image | User uploads a dermoscopic lesion image | Skin lesion agent |
| Brain MRI image | User uploads a brain MRI scan | Brain tumor agent or unsupported fallback |
| Ambiguous request | "Analyze this medical image" without image | Clarification or upload request |
| Unsafe request | "Tell me exact prescription dosage for my symptoms" | Guardrails / safety response |
| Out-of-scope | "Generate code for this app" | Guardrails or non-medical refusal |

## Unanswerable Question Strategy

The system handles insufficient information in layers:

1. If the question is unsafe or out of medical scope, input guardrails stop the flow.
2. If the question is stable medical knowledge, RAG retrieves from the local medical corpus.
3. If RAG confidence is below the configured threshold or the generated answer says the context is insufficient, the flow escalates to web search.
4. If web search or downstream tools fail, the answer states that available information is insufficient.
5. For image analysis, model outputs require human validation before being treated as accepted.

## Knowledge Base Design

The current repository contains 4 core documents in `data/raw` and a larger optional set in `data/raw_extras`. The README cites 12 medical papers or references covering:

- Brain tumor detection and deep learning methods
- Chest X-ray and COVID-19 diagnosis
- Diabetes mellitus
- Skin lesion analysis and segmentation

The knowledge base is organized by medical task domain:

| Domain | Example Documents | Retrieval Scope |
| --- | --- | --- |
| Brain tumor | Brain tumor introduction and detection papers | Tumor concepts, diagnosis, AI methods |
| Chest X-ray / COVID | COVID chest X-ray papers | Imaging findings and AI diagnosis research |
| Skin lesion | ISIC and lesion segmentation papers | Dermatology image analysis |
| General medical references | Diabetes and other optional materials | General medical facts if ingested |

Each document is parsed into text, tables, page images, and figure images. Figures are summarized by an LLM, combined into markdown content, semantically chunked, embedded, and stored in Qdrant with source metadata.
