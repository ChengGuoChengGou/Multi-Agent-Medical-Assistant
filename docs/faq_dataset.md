# Patient FAQ Dataset

## Purpose

The FAQ dataset complements the PDF-heavy medical knowledge base with patient-facing questions. The PDF layer is better for literature-grounded answers, while the FAQ layer improves recall for real user wording such as "Is my mole dangerous?" or "Can a chest X-ray confirm COVID?".

Current scope:

- 150 FAQ entries
- 7 medical domains
- Official or professional medical sources only
- Per-entry priority, risk level, doctor-needed flag, source organization, source URL, and keywords

Current local ingestion status:

- FAQ markdown is available to the FAQ keyword channel at runtime, so all 150 entries are used by lexical FAQ retrieval
- The checked-in local Qdrant/docstore snapshot still contains the earlier 131 indexed chunks; rerun FAQ ingestion to rebuild the vector snapshot with all 150 FAQ entries
- Full local vector store snapshot currently has 4 indexed PDFs + the earlier FAQ set, producing 131 Qdrant points/docstore files
- `data/raw_extras/` contains 12 additional candidate PDFs that are not indexed yet
- Domain distribution: 20 entries each for brain tumor, chest X-ray/COVID, skin lesion, diabetes, and general medical FAQ; 25 entries each for cardiovascular and medication safety
- Priority distribution: P0=54, P1=62, P2=34

## File Layout

```text
data/faq/
  brain_tumor_faq.md          20 entries
  chest_xray_covid_faq.md     20 entries
  skin_lesion_faq.md          20 entries
  diabetes_faq.md             20 entries
  general_medical_faq.md      20 entries
  cardiovascular_faq.md        25 entries
  medication_safety_faq.md     25 entries
```

## Priority Scheme

| Priority | Meaning | Usage |
| --- | --- | --- |
| P0 | High-frequency or safety-critical | Red flags, urgent care, diagnostic boundaries, medication safety |
| P1 | Common explanation | Symptoms, tests, treatment overview, follow-up, lifestyle guidance |
| P2 | Supplemental knowledge | Prevention, AI limitations, screening, process questions |

## Source Policy

FAQ answers are not free-form invented medical advice. They are patient-facing paraphrases based on official or professional sources:

- MedlinePlus
- National Cancer Institute
- Centers for Disease Control and Prevention
- RadiologyInfo / ACR-RSNA
- American College of Radiology
- American Academy of Dermatology
- American Diabetes Association
- NIDDK
- American Cancer Society

Sources are stored per entry using:

```text
source_org
source_url
source_type
retrieved_at
```

## Domain Coverage

| Domain | Entries | Main use |
| --- | ---: | --- |
| brain_tumor | 20 | Symptoms, MRI/CT, biopsy, treatment overview, red flags |
| chest_xray_covid | 20 | COVID symptoms, chest X-ray limits, pneumonia imaging, testing boundaries |
| skin_lesion | 20 | ABCDE melanoma checks, mole changes, biopsy, skin cancer risk |
| diabetes | 20 | Symptoms, A1C, diagnosis, hypoglycemia, medication safety |
| general_medical | 20 | Fever, antibiotics, urgent symptoms, AI limitations, medication safety |
| cardiovascular | 25 | Chest pain, stroke symptoms, blood pressure, palpitations, heart-risk triage |
| medication_safety | 25 | Prescription dose boundaries, overdose, OTC labels, interactions, allergy red flags |

## Inventory Command

Run:

```bash
python scripts/knowledge_inventory.py --json-output data/knowledge_inventory.json
```

Current output summary:

```text
Indexed PDFs: 4
Candidate PDFs not indexed: 12
FAQ markdown files: 7
FAQ entries: 150
Qdrant points: 131
Docstore files: 131
Parsed doc files: 194
```

## Interview Talking Point

> I found that the original RAG knowledge base was too PDF-heavy and too academic. Real users do not ask in paper titles; they ask patient-style questions. I added a 150-entry FAQ layer sourced from MedlinePlus, NCI, CDC, ACR/RadiologyInfo, AAD, ADA, NIDDK, ACS, AHA, the American Stroke Association, and FDA patient-safety material. Each FAQ has priority, domain, intent, risk level, doctor-needed flag, source URL, and keywords. This improves recall for colloquial medical queries and gives the system safer behavior around red flags and medication boundaries.

## Next Step

FAQ ingestion is wired into the standard local Qdrant/docstore path. Run:

```bash
python ingest_rag_data.py --faq-dir data/faq
```

Each FAQ entry becomes one searchable chunk and is tagged with:

```text
source_type=patient_faq
domain=<medical_domain>
priority=P0/P1/P2
risk_level=low/medium/high
```

The retriever now applies a lightweight FAQ priority boost before reranking, so P0 safety FAQ chunks are favored when the user asks about urgent symptoms, medication dosage, or diagnostic limits.

If the configured OpenAI-compatible provider does not expose an embeddings endpoint, enable the deterministic local fallback:

```text
USE_LOCAL_HASHING_EMBEDDINGS=true
EMBEDDING_DIM=1536
```

This is intended for local demos and interview validation. Production deployments should use a medical-quality embedding model exposed by the provider or a managed embedding service.
