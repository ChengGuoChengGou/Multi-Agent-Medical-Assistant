"""
Unit tests for agents.memory_module – InMemoryStore.

Tests cover:
  - Initialization (empty store)
  - remember: basic, multiple users, metadata
  - recall: empty, with content, keyword filter, limit
  - get_history: empty, with content, limit
  - forget: existing user, nonexistent, isolation

InMemoryStore is a pure dict-based fallback store with no external deps.
"""
import pytest

from agents.memory_module import InMemoryStore

# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def populated_store():
    s = InMemoryStore()
    s.remember("u1", "Headache for 3 days", {"symptom": True})
    s.remember("u1", "Took ibuprofen 400mg", {"medication": True})
    s.remember("u1", "Pain reduced after medication")
    s.remember("u2", "Allergic to penicillin", {"allergy": True})
    return s


# ─────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────

class TestInMemoryStoreInit:

    def test_new_store_empty(self, store):
        assert len(store._store) == 0

    def test_has_methods(self, store):
        assert callable(store.remember)
        assert callable(store.recall)
        assert callable(store.get_history)
        assert callable(store.forget)


# ─────────────────────────────────────────
# remember()
# ─────────────────────────────────────────

class TestRemember:

    def test_returns_bool(self, store):
        result = store.remember("user1", "test content")
        assert isinstance(result, bool)

    def test_creates_user_entry(self, store):
        store.remember("u1", "content")
        assert "u1" in store._store
        assert len(store._store["u1"]) == 1

    def test_stores_content(self, store):
        store.remember("u1", "headache symptoms")
        entry = store._store["u1"][0]
        assert entry["content"] == "headache symptoms"

    def test_stores_metadata(self, store):
        store.remember("u1", "content", {"key": "val"})
        entry = store._store["u1"][0]
        assert entry["metadata"] == {"key": "val"}

    def test_default_metadata_empty(self, store):
        store.remember("u1", "content")
        entry = store._store["u1"][0]
        assert entry["metadata"] == {}

    def test_appends_to_existing(self, store):
        store.remember("u1", "first")
        store.remember("u1", "second")
        assert len(store._store["u1"]) == 2
        assert store._store["u1"][1]["content"] == "second"

    def test_multiple_users_isolated(self, store):
        store.remember("u1", "user1 data")
        store.remember("u2", "user2 data")
        assert len(store._store["u1"]) == 1
        assert len(store._store["u2"]) == 1
        assert store._store["u1"][0]["content"] != store._store["u2"][0]["content"]


# ─────────────────────────────────────────
# recall()
# ─────────────────────────────────────────

class TestRecall:

    def test_empty_store_returns_empty_string(self, store):
        result = store.recall("u1", "anything")
        assert result == ""

    def test_empty_query_returns_empty_string(self, store):
        store.remember("u1", "some content")
        result = store.recall("u1", "")
        # Empty query → no keyword match → ""
        assert result == ""

    def test_returns_string(self, populated_store):
        result = populated_store.recall("u1", "headache")
        assert isinstance(result, str)

    def test_keyword_match_returns_prefixed_string(self, populated_store):
        result = populated_store.recall("u1", "headache")
        assert result != ""
        assert "Previously known information:" in result

    def test_keyword_match_contains_content(self, populated_store):
        result = populated_store.recall("u1", "headache")
        assert "headache" in result.lower()

    def test_nonexistent_keyword_returns_empty(self, populated_store):
        result = populated_store.recall("u1", "xyzzynotexist")
        assert result == ""

    def test_limit_parameter(self, populated_store):
        # Store has 3 entries for u1
        result = populated_store.recall("u1", "pain", limit=1)
        # Should return at most 1 match
        if result:  # If any match
            lines = [l for l in result.split("\n") if l.startswith("- ")]
            assert len(lines) <= 1

    def test_nonexistent_user_returns_empty(self, store):
        result = store.recall("nobody", "query")
        assert result == ""

    def test_medication_keyword(self, populated_store):
        result = populated_store.recall("u1", "ibuprofen")
        assert result != ""
        assert "ibuprofen" in result.lower()


# ─────────────────────────────────────────
# get_history()
# ─────────────────────────────────────────

class TestGetHistory:

    def test_empty_store_returns_empty_list(self, store):
        result = store.get_history("u1")
        assert result == []

    def test_returns_list(self, populated_store):
        result = populated_store.get_history("u1")
        assert isinstance(result, list)

    def test_returns_content_strings(self, populated_store):
        result = populated_store.get_history("u1")
        for item in result:
            assert isinstance(item, str)

    def test_all_entries_returned(self, populated_store):
        result = populated_store.get_history("u1")
        assert len(result) == 3

    def test_content_order_preserved(self, populated_store):
        result = populated_store.get_history("u1")
        assert result[0] == "Headache for 3 days"
        assert result[1] == "Took ibuprofen 400mg"
        assert result[2] == "Pain reduced after medication"

    def test_limit_parameter(self, populated_store):
        result = populated_store.get_history("u1", limit=2)
        assert len(result) == 2
        # Last 2 entries
        assert result[0] == "Took ibuprofen 400mg"
        assert result[1] == "Pain reduced after medication"

    def test_limit_larger_than_store(self, populated_store):
        result = populated_store.get_history("u1", limit=100)
        assert len(result) == 3

    def test_nonexistent_user_returns_empty_list(self, store):
        result = store.get_history("nobody")
        assert result == []

    def test_other_user_not_affected(self, populated_store):
        result = populated_store.get_history("u2")
        assert len(result) == 1
        assert result[0] == "Allergic to penicillin"


# ─────────────────────────────────────────
# forget()
# ─────────────────────────────────────────

class TestForget:

    def test_returns_bool(self, populated_store):
        result = populated_store.forget("u1")
        assert isinstance(result, bool)

    def test_removes_user(self, populated_store):
        populated_store.forget("u1")
        assert "u1" not in populated_store._store

    def test_other_users_preserved(self, populated_store):
        populated_store.forget("u1")
        assert "u2" in populated_store._store
        assert len(populated_store._store["u2"]) == 1

    def test_nonexistent_user_returns_bool(self, store):
        result = store.forget("nobody")
        assert isinstance(result, bool)

    def test_double_forget_safe(self, populated_store):
        populated_store.forget("u1")
        result = populated_store.forget("u1")
        assert isinstance(result, bool)

    def test_recall_after_forget_returns_empty(self, populated_store):
        populated_store.forget("u1")
        result = populated_store.recall("u1", "headache")
        assert result == ""

    def test_get_history_after_forget_returns_empty(self, populated_store):
        populated_store.forget("u1")
        result = populated_store.get_history("u1")
        assert result == []

    def test_remember_after_forget_works(self, populated_store):
        populated_store.forget("u1")
        populated_store.remember("u1", "new memory")
        result = populated_store.get_history("u1")
        assert len(result) == 1
        assert result[0] == "new memory"


# ─────────────────────────────────────────
# Integration / Lifecycle
# ─────────────────────────────────────────

class TestLifecycle:

    def test_full_lifecycle(self, store):
        # Remember
        store.remember("patient1", "Headache and fever", {"symptom": True})
        store.remember("patient1", "Prescribed paracetamol", {"medication": True})

        # Recall
        result = store.recall("patient1", "headache")
        assert result != ""
        assert "Previously known information:" in result

        # History
        history = store.get_history("patient1")
        assert len(history) == 2

        # Add more
        store.remember("patient1", "Follow-up: symptoms resolved")
        assert len(store.get_history("patient1")) == 3

        # Recall with new info
        result = store.recall("patient1", "symptoms")
        assert "symptoms" in result.lower()

        # Forget
        store.forget("patient1")
        assert store.get_history("patient1") == []
        assert store.recall("patient1", "headache") == ""

    def test_multiple_patients(self, store):
        store.remember("p1", "Patient 1: diabetes", {})
        store.remember("p2", "Patient 2: hypertension", {})
        store.remember("p3", "Patient 3: asthma", {})

        assert len(store.get_history("p1")) == 1
        assert len(store.get_history("p2")) == 1
        assert len(store.get_history("p3")) == 1

        store.forget("p2")
        assert len(store.get_history("p1")) == 1
        assert store.get_history("p2") == []
        assert len(store.get_history("p3")) == 1
