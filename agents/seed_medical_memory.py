r"""
Seed Medical Vector Memory with core medical knowledge.
Run once to populate the vector store with essential medical facts.

Usage:
    python -m agents.seed_medical_memory
    or
    cd D:\Code\Multi-Agent-Medical-Assistant
    python agents/seed_medical_memory.py
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.medical_vector_memory import add_memory, collection_stats


def seed_medical_facts():
    """Seed the vector memory with core medical knowledge."""

    # === Core Medical Knowledge ===
    medical_facts = [
        # Symptoms & Conditions
        (
            "Hypertension (high blood pressure) is defined as systolic ≥130 mmHg or diastolic ≥80 mmHg. "
            "Often called the 'silent killer' because it typically has no symptoms until serious damage occurs.",
            {"type": "medical_fact", "category": "cardiovascular", "topic": "hypertension"},
        ),
        (
            "Type 2 diabetes is characterized by insulin resistance. Key symptoms include polyuria (frequent urination), "
            "polydipsia (excessive thirst), polyphagia (increased hunger), and unexplained weight loss.",
            {"type": "medical_fact", "category": "endocrine", "topic": "diabetes"},
        ),
        (
            "Common symptoms of COVID-19 include fever, dry cough, fatigue, loss of taste/smell, sore throat, "
            "headache, body aches, and difficulty breathing in severe cases.",
            {"type": "medical_fact", "category": "infectious", "topic": "covid19"},
        ),
        (
            "Asthma is a chronic lung disease characterized by inflamed airways. Triggers include allergens, "
            "cold air, exercise, and respiratory infections. Treatment includes inhalers (bronchodilators and corticosteroids).",
            {"type": "medical_fact", "category": "respiratory", "topic": "asthma"},
        ),
        (
            "Pneumonia is an infection that inflames air sacs in one or both lungs. Symptoms include chest pain, "
            "cough with phlegm, fever, chills, and difficulty breathing. Can be bacterial, viral, or fungal.",
            {"type": "medical_fact", "category": "respiratory", "topic": "pneumonia"},
        ),
        # Medications
        (
            "Metformin is a first-line medication for type 2 diabetes. It decreases hepatic glucose production "
            "and increases insulin sensitivity. Common side effects include GI upset. Contraindicated in severe renal impairment.",
            {"type": "medical_fact", "category": "medication", "topic": "metformin"},
        ),
        (
            "ACE inhibitors (e.g., lisinopril, enalapril) are used for hypertension and heart failure. "
            "Common side effect is dry cough. Contraindicated in pregnancy. Monitor potassium and renal function.",
            {"type": "medical_fact", "category": "medication", "topic": "ace_inhibitors"},
        ),
        (
            "Aspirin (acetylsalicylic acid) is used as an antiplatelet agent for cardiovascular prevention. "
            "Low-dose (81mg) for secondary prevention. Risk of GI bleeding. Not recommended for children (Reye's syndrome).",
            {"type": "medical_fact", "category": "medication", "topic": "aspirin"},
        ),
        # Emergency Signs
        (
            "Chest pain with shortness of breath, sweating, nausea, and pain radiating to left arm or jaw "
            "may indicate a heart attack (myocardial infarction). Call emergency services immediately.",
            {"type": "medical_fact", "category": "emergency", "topic": "heart_attack"},
        ),
        (
            "Stroke warning signs (FAST): Face drooping, Arm weakness, Speech difficulty, Time to call emergency. "
            "Sudden severe headache, confusion, vision problems, or dizziness may also indicate stroke.",
            {"type": "medical_fact", "category": "emergency", "topic": "stroke"},
        ),
        # General Health
        (
            "Normal adult vital signs: Heart rate 60-100 bpm, Blood pressure <120/80 mmHg, "
            "Respiratory rate 12-20 breaths/min, Temperature 97.8-99.1°F (36.5-37.3°C).",
            {"type": "medical_fact", "category": "general", "topic": "vital_signs"},
        ),
        (
            "BMI categories: Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, "
            "Obesity ≥30. BMI = weight(kg) / height(m)².",
            {"type": "medical_fact", "category": "general", "topic": "bmi"},
        ),
        # Mental Health
        (
            "Major Depressive Disorder (MDD) requires ≥2 weeks of depressed mood or loss of interest, "
            "plus symptoms like weight changes, sleep disturbance, fatigue, guilt, and concentration problems.",
            {"type": "medical_fact", "category": "psychiatric", "topic": "depression"},
        ),
        (
            "Generalized Anxiety Disorder (GAD) involves excessive worry about multiple things for ≥6 months, "
            "with restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep problems.",
            {"type": "medical_fact", "category": "psychiatric", "topic": "anxiety"},
        ),
        # Nutritional
        (
            "Recommended daily water intake: ~3.7L for men, ~2.7L for women (including from food). "
            "Increase with exercise, hot weather, illness, or pregnancy.",
            {"type": "medical_fact", "category": "nutrition", "topic": "hydration"},
        ),
    ]

    # === Project & Interaction Patterns ===
    interaction_facts = [
        (
            "用户偏好简洁有结构的回答，使用Markdown格式。重要信息加粗。列表用编号。技术问题直接给方案，不重复背景。",
            {"type": "user_pref", "lang": "zh", "category": "communication"},
        ),
        (
            "This medical assistant uses a multi-agent architecture with specialized agents: "
            "Conversation agent (chat), RAG agent (knowledge retrieval), Chest X-ray agent (image analysis), "
            "and Web Search agent (real-time info). Routed by a decision agent.",
            {"type": "project_fact", "category": "architecture"},
        ),
        (
            "The system implements Phase 51 conversation management with LLM-based summarization. "
            "Old messages are summarized when conversation exceeds max_conversation_history limit.",
            {"type": "project_fact", "category": "memory_management"},
        ),
    ]

    # Seed medical facts
    print("🏥 Seeding medical vector memory...")
    all_facts = medical_facts + interaction_facts
    success = 0
    fail = 0

    for text, metadata in all_facts:
        try:
            result = add_memory(text=text, metadata=metadata)
            if result:
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ⚠ Failed to seed: {text[:50]}... ({e})")
            fail += 1

    # Print stats
    stats = collection_stats()
    print(f"\n✅ Seeding complete: {success} added, {fail} skipped")
    print(f"📊 Collection stats: {stats}")

    return success, fail


if __name__ == "__main__":
    seed_medical_facts()
