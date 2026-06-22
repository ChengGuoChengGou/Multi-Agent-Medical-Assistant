"""
Unit tests for tools/registry.py — ToolRegistry singleton, tool management.
Run: python -m pytest tests/test_registry.py -v
"""
import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry, get_registry


# ── Helpers ──

def _reset_singleton():
    """Reset ToolRegistry singleton state for clean tests."""
    ToolRegistry._instance = None
    ToolRegistry._initialized = False


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset singleton before each test."""
    _reset_singleton()
    import tools.registry as mod
    mod._registry_instance = None
    yield
    _reset_singleton()
    mod._registry_instance = None


# ── Singleton behavior ──
class TestSingleton:
    def test_singleton_returns_same_instance(self):
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        assert r1 is r2

    def test_get_registry_returns_tool_registry(self):
        r = get_registry()
        assert isinstance(r, ToolRegistry)

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_get_registry_matches_direct(self):
        r1 = get_registry()
        r2 = ToolRegistry()
        assert r1 is r2


# ── initialize ──
class TestInitialize:
    def test_initialize_no_medical_module(self, monkeypatch):
        """When agents.medical_tool is not importable, returns False."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "agents.medical_tool":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        r = ToolRegistry()
        result = r.initialize()
        assert result is False

    def test_initialize_success(self, monkeypatch):
        """When medical_tool imports OK, returns True."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "agents.medical_tool":
                class FakeModule:
                    def init_tool_registry(self): pass
                    def get_tool_registry(self): return {}
                return FakeModule()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        r = ToolRegistry()
        result = r.initialize()
        assert result is True


# ── get_tool ──
class TestGetTool:
    def test_get_tool_no_registry(self):
        r = ToolRegistry()
        r._medical_registry = None
        assert r.get_tool("anything") is None

    def test_get_tool_from_external(self):
        r = ToolRegistry()
        r._medical_registry = None
        r._external_tools = {"my_tool": "tool_obj"}
        assert r.get_tool("my_tool") == "tool_obj"

    def test_get_tool_external_default(self):
        r = ToolRegistry()
        r._medical_registry = None
        assert r.get_tool("nonexistent") is None


# ── register_external_tool ──
class TestRegisterExternalTool:
    def test_register_and_retrieve(self):
        r = ToolRegistry()
        r._medical_registry = None
        tool = type("FakeTool", (), {"execute": lambda self: 42})()
        r.register_external_tool("test_tool", tool)
        assert r.get_tool("test_tool") is tool

    def test_register_overwrites(self):
        r = ToolRegistry()
        r._medical_registry = None
        r.register_external_tool("dup", "first")
        r.register_external_tool("dup", "second")
        assert r.get_tool("dup") == "second"


# ── get_summary ──
class TestGetSummary:
    def test_no_tools(self):
        r = ToolRegistry()
        r._medical_registry = None
        assert r.get_summary() == "No tools available"

    def test_external_tools_only(self):
        r = ToolRegistry()
        r._medical_registry = None
        tool = type("T", (), {"description": "A test tool"})()
        r.register_external_tool("ext_tool", tool)
        summary = r.get_summary()
        assert "External Tools:" in summary
        assert "ext_tool" in summary
        assert "A test tool" in summary

    def test_max_desc_len(self):
        r = ToolRegistry()
        r._medical_registry = None
        tool = type("T", (), {"description": "A" * 200})()
        r.register_external_tool("long_desc", tool)
        summary = r.get_summary(max_desc_len=50)
        assert "..." in summary


# ── get_all_tool_names ──
class TestGetAllToolNames:
    def test_no_tools(self):
        r = ToolRegistry()
        r._medical_registry = None
        assert r.get_all_tool_names() == []

    def test_external_names(self):
        r = ToolRegistry()
        r._medical_registry = None
        r.register_external_tool("a", "obj_a")
        r.register_external_tool("b", "obj_b")
        names = r.get_all_tool_names()
        assert "a" in names
        assert "b" in names


# ── is_available ──
class TestIsAvailable:
    def test_no_tools(self):
        r = ToolRegistry()
        r._medical_registry = None
        assert r.is_available() is False

    def test_external_tools_available(self):
        r = ToolRegistry()
        r._medical_registry = None
        r.register_external_tool("t", "obj")
        assert r.is_available() is True
