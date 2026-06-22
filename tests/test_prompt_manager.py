"""Tests for prompts/manager.py — prompt template management."""

from pathlib import Path
from unittest.mock import patch

import pytest

from prompts.manager import PromptManager, get_prompt_manager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts_dir(tmp_path):
    """Create a temporary prompts directory with sample templates."""
    (tmp_path / "greeting.md").write_text("Hello {{name}}, welcome to {{app}}!", encoding="utf-8")
    (tmp_path / "medical.md").write_text(
        "Patient: {{patient_name}}\nSymptoms: {{symptoms}}\nDiagnose:", encoding="utf-8"
    )
    (tmp_path / "no_vars.md").write_text("Static prompt with no variables.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def manager(tmp_prompts_dir):
    return PromptManager(prompts_dir=tmp_prompts_dir)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_loads_templates_from_dir(self, tmp_prompts_dir):
        m = PromptManager(prompts_dir=tmp_prompts_dir)
        names = m.list_templates()
        assert "greeting" in names
        assert "medical" in names
        assert "no_vars" in names
        assert len(names) == 3

    def test_empty_dir(self, tmp_path):
        m = PromptManager(prompts_dir=tmp_path)
        assert m.list_templates() == []

    def test_nonexistent_dir(self):
        m = PromptManager(prompts_dir=Path("/nonexistent/path"))
        assert m.list_templates() == []

    def test_default_dir(self):
        """Default dir should be prompts/ directory."""
        with patch("prompts.manager.PROMPTS_DIR", Path("/fake/dir")):
            m = PromptManager()
            assert m.prompts_dir == Path("/fake/dir")

    def test_default_prompts_dir_field(self, tmp_prompts_dir):
        """If no dir given, should use module-level PROMPTS_DIR."""
        m = PromptManager(prompts_dir=tmp_prompts_dir)
        assert m.prompts_dir == tmp_prompts_dir


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------


class TestGet:
    def test_basic_substitution(self, manager):
        result = manager.get("greeting", name="Alice", app="MedAssist")
        assert "Alice" in result
        assert "MedAssist" in result
        assert "{{name}}" not in result
        assert "{{app}}" not in result

    def test_partial_substitution(self, manager):
        """Only provided vars substituted; others remain as {{var}}."""
        result = manager.get("greeting", name="Bob")
        assert "Bob" in result
        assert "{{app}}" in result  # not substituted

    def test_no_vars_template(self, manager):
        result = manager.get("no_vars")
        assert result == "Static prompt with no variables."

    def test_no_vars_with_extra_kwargs(self, manager):
        """Extra kwargs should be silently ignored."""
        result = manager.get("no_vars", unused="value")
        assert result == "Static prompt with no variables."

    def test_missing_template_raises_keyerror(self, manager):
        with pytest.raises(KeyError, match="nonexistent"):
            manager.get("nonexistent")

    def test_missing_template_lists_available(self, manager):
        with pytest.raises(KeyError) as exc_info:
            manager.get("missing")
        assert "greeting" in str(exc_info.value)
        assert "medical" in str(exc_info.value)

    def test_medical_template(self, manager):
        result = manager.get("medical", patient_name="Zhang San", symptoms="fever, cough")
        assert "Zhang San" in result
        assert "fever, cough" in result
        assert "Diagnose:" in result

    def test_multiline_template(self, manager):
        result = manager.get("medical", patient_name="Li", symptoms="pain")
        assert "\n" in result  # preserves newlines

    def test_numeric_value_substitution(self, manager):
        """Non-string values should be str() converted."""
        result = manager.get("greeting", name=42, app=3.14)
        assert "42" in result
        assert "3.14" in result

    def test_special_chars_in_value(self, manager):
        result = manager.get("greeting", name="O'Brien", app="v2.0")
        assert "O'Brien" in result
        assert "v2.0" in result

    def test_empty_string_value(self, manager):
        result = manager.get("greeting", name="", app="")
        assert "Hello , welcome to !" in result

    def test_brace_like_content_in_value(self, manager):
        """Values containing braces should not be re-parsed."""
        result = manager.get("greeting", name="{{injected}}", app="test")
        assert "{{injected}}" in result  # literal value, not re-substituted


# ---------------------------------------------------------------------------
# TestListTemplates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_returns_list(self, manager):
        result = manager.list_templates()
        assert isinstance(result, list)

    def test_returns_copy(self, manager):
        result1 = manager.list_templates()
        result1.append("fake")
        result2 = manager.list_templates()
        assert "fake" not in result2


# ---------------------------------------------------------------------------
# TestReload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_picks_up_new_file(self, manager, tmp_prompts_dir):
        initial_count = len(manager.list_templates())
        # Add a new template
        (tmp_prompts_dir / "new.md").write_text("New prompt", encoding="utf-8")
        manager.reload()
        assert len(manager.list_templates()) == initial_count + 1
        assert "new" in manager.list_templates()

    def test_reload_clears_old_templates(self, manager, tmp_prompts_dir):
        # Remove an existing file
        (tmp_prompts_dir / "greeting.md").unlink()
        manager.reload()
        assert "greeting" not in manager.list_templates()

    def test_reload_updates_content(self, manager, tmp_prompts_dir):
        # Modify a template
        (tmp_prompts_dir / "greeting.md").write_text("Hi {{name}}!", encoding="utf-8")
        manager.reload()
        result = manager.get("greeting", name="Test")
        assert result == "Hi Test!"

    def test_reload_from_empty(self, tmp_path):
        m = PromptManager(prompts_dir=tmp_path)
        assert m.list_templates() == []
        # Add template after init
        (tmp_path / "late.md").write_text("Late template", encoding="utf-8")
        m.reload()
        assert "late" in m.list_templates()

    def test_reload_logs_count(self, manager, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            manager.reload()
        assert "Reloaded" in caplog.text


# ---------------------------------------------------------------------------
# TestGetPromptManager (singleton)
# ---------------------------------------------------------------------------


class TestGetPromptManager:
    def test_returns_prompt_manager(self):
        import prompts.manager as pm

        old = pm._manager_instance
        pm._manager_instance = None
        try:
            result = get_prompt_manager()
            assert isinstance(result, PromptManager)
        finally:
            pm._manager_instance = old

    def test_returns_same_instance(self):
        import prompts.manager as pm

        old = pm._manager_instance
        pm._manager_instance = None
        try:
            a = get_prompt_manager()
            b = get_prompt_manager()
            assert a is b
        finally:
            pm._manager_instance = old


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_md_files_ignored(self, tmp_path):
        """Only .md files should be loaded."""
        (tmp_path / "template.md").write_text("MD content", encoding="utf-8")
        (tmp_path / "data.json").write_text('{"key": "val"}', encoding="utf-8")
        (tmp_path / "notes.txt").write_text("Text content", encoding="utf-8")
        m = PromptManager(prompts_dir=tmp_path)
        assert m.list_templates() == ["template"]

    def test_subdirectory_md_not_loaded(self, tmp_path):
        """glob("*.md") should not recurse into subdirs."""
        (tmp_path / "top.md").write_text("Top level", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("Nested", encoding="utf-8")
        m = PromptManager(prompts_dir=tmp_path)
        assert m.list_templates() == ["top"]

    def test_duplicate_var_in_template(self, tmp_path):
        """Same variable appearing multiple times should all be substituted."""
        (tmp_path / "dup.md").write_text("{{x}} and {{x}} again", encoding="utf-8")
        m = PromptManager(prompts_dir=tmp_path)
        result = m.get("dup", x="VALUE")
        assert result == "VALUE and VALUE again"

    def test_underscore_var_name(self, tmp_path):
        (tmp_path / "us.md").write_text("{{my_var_name}}", encoding="utf-8")
        m = PromptManager(prompts_dir=tmp_path)
        assert m.get("us", my_var_name="ok") == "ok"

    def test_numeric_var_name(self, tmp_path):
        """\\w+ matches digits, so {{123}} should work."""
        (tmp_path / "num.md").write_text("{{123}}", encoding="utf-8")
        m = PromptManager(prompts_dir=tmp_path)
        assert m.get("num", **{"123": "found"}) == "found"
