"""Quick test: verify medical_vector_memory search works."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.medical_vector_memory import collection_stats, search_memory

stats = collection_stats()
print("Collection stats:", stats)

# search_memory returns list of text strings
results = search_memory("What is diabetes?", min_score=0.3, top_k=3)
print("\nSearch 'diabetes' returned %d results:" % len(results))
for i, r in enumerate(results):
    print("  %d. %s..." % (i + 1, r[:120]))

results2 = search_memory("high blood pressure treatment", min_score=0.3, top_k=3)
print("\nSearch 'hypertension' returned %d results:" % len(results2))
for i, r in enumerate(results2):
    print("  %d. %s..." % (i + 1, r[:120]))

results3 = search_memory("headache nausea causes", min_score=0.3, top_k=3)
print("\nSearch 'headache nausea' returned %d results:" % len(results3))
for i, r in enumerate(results3):
    print("  %d. %s..." % (i + 1, r[:120]))

print("\nDone!")
