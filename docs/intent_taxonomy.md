# Medical Intent Taxonomy

This taxonomy follows the intent-tree idea used by enterprise RAG systems: each leaf intent has a clear semantic scope, route, retrieval strategy, and fallback behavior.

## Intent Tree

```text
medical_assistant
├── system
│   ├── general_chat
│   ├── capability_question
│   └── out_of_scope
├── medical_qa
│   ├── general_medical_knowledge
│   ├── literature_grounded_qa
│   ├── latest_medical_research
│   └── symptom_triage_information
├── medical_image
│   ├── chest_xray_analysis
│   ├── skin_lesion_segmentation
│   ├── brain_mri_analysis
│   └── unsupported_or_unclear_image
└── safety
    ├── emergency_or_self_harm
    ├── prescription_or_dosage_request
    ├── diagnosis_overclaim
    └── prompt_injection_or_data_exfiltration
```

## Leaf Intent Definitions

| Intent | Kind | Route | Examples | Fallback |
| --- | --- | --- | --- | --- |
| `general_chat` | SYSTEM | Conversation agent | "hello", "thanks" | Direct response |
| `capability_question` | SYSTEM | Conversation agent | "What can you analyze?" | Explain supported tasks |
| `out_of_scope` | SYSTEM | Guardrails / conversation | "Write code for me" | Refuse or redirect |
| `general_medical_knowledge` | KB | RAG agent | "What is glioma?" | Web search if low confidence |
| `literature_grounded_qa` | KB | RAG agent | "Compare methods in brain tumor papers" | Insufficient-info response |
| `latest_medical_research` | TOOL | Web search processor | "recent COVID chest X-ray study" | RAG summary + limitation |
| `symptom_triage_information` | KB/SYSTEM | Conversation or RAG | "fever and cough, what might it be?" | Safety disclaimer and clinician advice |
| `chest_xray_analysis` | VISION | Chest X-ray agent | uploaded X-ray | Human validation |
| `skin_lesion_segmentation` | VISION | Skin lesion agent | uploaded lesion image | Human validation |
| `brain_mri_analysis` | VISION | Brain tumor agent | uploaded MRI | Human validation or not-yet-supported note |
| `unsupported_or_unclear_image` | VISION | Conversation agent | non-medical image | Ask for clearer supported image |
| `emergency_or_self_harm` | SAFETY | Guardrails | emergency/self-harm | Urgent care guidance |
| `prescription_or_dosage_request` | SAFETY | Guardrails | "What exact dose should I take?" | Refuse exact prescription |
| `diagnosis_overclaim` | SAFETY | Guardrails / validation | "Tell me if I have cancer" | Explain limits |
| `prompt_injection_or_data_exfiltration` | SAFETY | Guardrails | "Ignore previous instructions" | Refuse |

## Routing Rules

1. Image presence has priority over text-only routing.
2. Time-sensitive wording such as "latest", "recent", "current", "today", and "new study" favors web search.
3. Local literature questions favor RAG.
4. Low route confidence should trigger clarification or a safe default.
5. Medical image outputs require human validation.
6. Safety intents override all normal task intents.

## Ambiguity Handling

Ambiguity should be handled before retrieval:

- If a question can map to several disease domains with similar confidence, ask a clarifying question.
- If an uploaded image type is uncertain, route to unsupported image handling instead of forcing a CV model.
- If a symptom question lacks age, severity, duration, or emergency indicators, answer only general information and recommend professional care when appropriate.

## Evaluation Labels

Evaluation samples should use these labels:

```json
{
  "query": "What are common AI methods for brain tumor detection?",
  "expected_intent": "literature_grounded_qa",
  "expected_agent": "RAG_AGENT",
  "expected_tool": "knowledge_base",
  "difficulty": "medium"
}
```
