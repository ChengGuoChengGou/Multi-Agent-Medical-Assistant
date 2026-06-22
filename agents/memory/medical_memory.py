"""
Medical Memory Interface - Healthcare-specific memory operations.

Provides specialized memory operations for medical context:
- Patient history tracking
- Symptom correlation memory
- Medication allergy persistence
- Clinical context recall with medical relevance scoring
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MedicalMemory:
    """
    Medical-specialized memory interface.
    
    Wraps the base MemoryStore with healthcare-specific operations:
    - Tagged medical categories (symptoms, medications, diagnoses)
    - Patient context persistence
    - Medical relevance filtering
    """
    
    # Medical memory categories
    CATEGORIES = {
        "symptom": "Patient-reported symptoms",
        "medication": "Medications and prescriptions",
        "allergy": "Known allergies and adverse reactions",
        "diagnosis": "Diagnostic conclusions",
        "history": "Patient medical history",
        "lifestyle": "Lifestyle and environmental factors",
    }
    
    def __init__(self):
        try:
            from agents.memory_module import get_memory_store
            self._store = get_memory_store()
            self._available = True
            logger.info("[MEDICAL_MEMORY] MedicalMemory initialized")
        except Exception as e:
            logger.warning(f"[MEDICAL_MEMORY] Init failed: {e}")
            self._store = None
            self._available = False
    
    @property
    def available(self) -> bool:
        return self._available and self._store is not None
    
    def remember_medical(self, user_id: str, content: str, 
                         category: str = "general",
                         metadata: Dict[str, Any] = None) -> bool:
        """
        Store medical information with category tagging.
        
        Args:
            user_id: Patient/user identifier
            content: Medical information to store
            category: One of CATEGORIES keys (symptom/medication/allergy/diagnosis/history/lifestyle)
            metadata: Additional metadata (e.g., severity, date, source)
            
        Returns:
            True if stored successfully
        """
        if not self.available:
            logger.warning("[MEDICAL_MEMORY] Memory store not available")
            return False
        
        try:
            meta = metadata or {}
            meta["category"] = category
            meta["medical"] = True
            
            return self._store.remember(user_id, content, meta)
        except Exception as e:
            logger.error(f"[MEDICAL_MEMORY] Remember failed: {e}")
            return False
    
    def recall_medical(self, user_id: str, query: str, 
                       category: str = None,
                       limit: int = 5) -> str:
        """
        Recall medical context relevant to a query.
        
        Args:
            user_id: Patient/user identifier
            query: Current query or topic
            category: Optional filter by medical category
            limit: Maximum results
            
        Returns:
            Formatted string of relevant medical memories
        """
        if not self.available:
            return ""
        
        try:
            # Enhance query with medical context
            enhanced_query = query
            if category:
                enhanced_query = f"[{category}] {query}"
            
            return self._store.recall(user_id, enhanced_query, limit=limit)
        except Exception as e:
            logger.error(f"[MEDICAL_MEMORY] Recall failed: {e}")
            return ""
    
    def get_patient_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """
        Get patient's medical history as structured records.
        
        Args:
            user_id: Patient/user identifier
            limit: Maximum history entries
            
        Returns:
            List of medical history records
        """
        if not self.available:
            return []
        
        try:
            return self._store.get_history(user_id, limit=limit)
        except Exception as e:
            logger.error(f"[MEDICAL_MEMORY] Get history failed: {e}")
            return []
    
    def check_allergies(self, user_id: str) -> str:
        """Recall known allergies for a patient."""
        return self.recall_medical(user_id, "allergies medications adverse reactions", 
                                   category="allergy", limit=10)
    
    def check_medications(self, user_id: str) -> str:
        """Recall current medications for a patient."""
        return self.recall_medical(user_id, "current medications prescriptions", 
                                   category="medication", limit=10)
    
    def forget_patient(self, user_id: str) -> bool:
        """Remove all medical records for a patient (GDPR compliance)."""
        if not self.available:
            return False
        
        try:
            return self._store.forget(user_id)
        except Exception as e:
            logger.error(f"[MEDICAL_MEMORY] Forget failed: {e}")
            return False


# Singleton
_medical_memory: Optional[MedicalMemory] = None


def get_medical_memory() -> MedicalMemory:
    """Get singleton MedicalMemory instance."""
    global _medical_memory
    if _medical_memory is None:
        _medical_memory = MedicalMemory()
    return _medical_memory
