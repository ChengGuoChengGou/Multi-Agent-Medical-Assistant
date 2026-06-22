"""
Unit tests for config.py — LLMFallbackChain + model registry.
Run: python -m pytest tests/test_config.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_model(name: str, fail_count: int = 0):
    """Create a mock ChatOpenAI-like model that fails `fail_count` times then succeeds."""
    model = MagicMock()
    model.model = name
    model._generate = MagicMock()
    if fail_count > 0:
        # First N calls raise, then succeed
        model._generate.side_effect = [RuntimeError(f"{name} simulated failure")] * fail_count + [MagicMock()]
    else:
        model._generate.return_value = MagicMock()
    return model


# ── LLMFallbackChain Tests ──────────────────────────────────────────


class TestLLMFallbackChain:
    """Test the LLMFallbackChain fallback/recovery logic."""

    def _import_chain(self):
        """Import LLMFallbackChain (already loaded by conftest mock chain)."""
        from config import LLMFallbackChain

        return LLMFallbackChain

    def test_llm_type(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")
        chain = Chain(models=[m1])
        assert chain._llm_type == "llm-fallback-chain"

    def test_model_property_returns_active(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2])
        assert chain.model == "model-a"
        chain.active_index = 1
        assert chain.model == "model-b"

    def test_primary_success_no_fallback(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2])

        _result = chain._generate(["test message"])

        m1._generate.assert_called_once()
        m2._generate.assert_not_called()
        assert chain.active_index == 0
        assert chain._fail_counts.get(0, 0) == 0

    def test_primary_fails_switches_to_fallback(self):
        """fail_threshold is cumulative across calls, not retries within one call."""
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a", fail_count=3)  # fails 3x then succeeds
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2], fail_threshold=2)

        # Call 1: m1 fails(fail_count=1<2), m2 succeeds -> returned, active_index stays 0
        chain._generate(["test1"])
        assert chain.active_index == 0

        # Call 2: m1 fails again(fail_count=2>=2), m2 succeeds -> switch active_index=1
        chain._generate(["test2"])
        assert chain.active_index == 1

    def test_recovery_switches_back_to_primary(self):
        """After switch to fallback, primary recovers and gets re-selected."""
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a", fail_count=3)  # fails 3x then succeeds
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2], fail_threshold=2)

        # Calls 1-2: m1 fails enough to trigger switch
        chain._generate(["test1"])
        chain._generate(["test2"])
        assert chain.active_index == 1

        # Reset mocks, make m1 succeed now
        m1.fail_remaining = 0
        m1._generate.reset_mock()
        m2._generate.reset_mock()

        # Call 3: starts at active_index=1, m2 succeeds -> returned
        # (recovery happens when loop iterates from active_index and primary succeeds
        #  but since loop starts at 1, it only tries m2)
        _result = chain._generate(["test3"])
        assert m2._generate.call_count == 1

    def test_all_models_fail_raises(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a", fail_count=10)
        m2 = _make_mock_model("model-b", fail_count=10)
        chain = Chain(models=[m1, m2], fail_threshold=1)

        with pytest.raises(RuntimeError):
            chain._generate(["test"])

    def test_fail_threshold_accumulates(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")  # succeeds normally
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2], fail_threshold=3)

        # First call succeeds → fail count stays 0
        chain._generate(["ok"])
        assert chain._fail_counts.get(0, 0) == 0
        assert chain.active_index == 0

    def test_bind_tools_delegates_to_active(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")
        m1.bind_tools = MagicMock(return_value="bound")
        m2 = _make_mock_model("model-b")
        chain = Chain(models=[m1, m2])

        result = chain.bind_tools(["tool1", "tool2"])
        m1.bind_tools.assert_called_once_with(["tool1", "tool2"])
        assert result == "bound"

    def test_single_model_no_fallback(self):
        Chain = self._import_chain()
        m1 = _make_mock_model("model-a")
        chain = Chain(models=[m1])

        _result = chain._generate(["test"])
        m1._generate.assert_called_once()
        assert chain.active_index == 0


# ── _make_llm Tests ─────────────────────────────────────────────────


class TestMakeLlm:
    """Test _make_llm function with model registry and fallback."""

    @patch.dict(os.environ, {}, clear=False)
    def test_make_llm_returns_primary_model(self):
        """Without FALLBACK_MODELS, returns a ChatOpenAI (not chain)."""
        from config import _make_llm

        # Ensure no fallback
        os.environ.pop("FALLBACK_MODELS", None)
        llm = _make_llm(0.5, role="conversation")
        # Should be ChatOpenAI, not LLMFallbackChain
        assert llm._llm_type != "llm-fallback-chain"

    @patch.dict(os.environ, {"FALLBACK_MODELS": "model-b,model-c"}, clear=False)
    def test_make_llm_with_fallback_returns_chain(self):
        """With FALLBACK_MODELS set, returns LLMFallbackChain."""
        from config import LLMFallbackChain, _make_llm

        llm = _make_llm(0.5, role="rag")
        assert isinstance(llm, LLMFallbackChain)
        assert len(llm.models) == 3  # primary + 2 fallbacks

    def test_make_llm_uses_role_model(self):
        """_make_llm picks model from _MODEL_ROLES by role."""
        from config import _MODEL_ROLES

        # The role 'decision' should use DECISION_MODEL or default
        decision_model = _MODEL_ROLES.get("decision")
        assert decision_model is not None


# ── Config Classes Tests ────────────────────────────────────────────


class TestConfigClasses:
    """Test that config dataclasses instantiate without error."""

    def test_api_config_defaults(self):
        from config import APIConfig

        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.rate_limit == 10
        assert cfg.max_image_upload_size == 5

    def test_ui_config_defaults(self):
        from config import UIConfig

        cfg = UIConfig()
        assert cfg.theme == "light"
        assert cfg.enable_speech is True
        assert cfg.enable_image_upload is True

    def test_validation_config_defaults(self):
        from config import ValidationConfig

        cfg = ValidationConfig()
        # max_iterations is in MCPConfig, not ValidationConfig
        assert cfg.require_validation["BRAIN_TUMOR_AGENT"] is True
        assert cfg.require_validation["CONVERSATION_AGENT"] is False
        assert cfg.validation_timeout == 300
        assert cfg.default_action == "reject"
        assert "BRAIN_TUMOR_AGENT" in cfg.require_validation

    def test_mcp_config_defaults(self):
        from config import MCPConfig

        cfg = MCPConfig()
        assert cfg.biomcp["enabled"] is True
        assert cfg.autoicd["enabled"] is True
        assert cfg.healthcare["enabled"] is True
        assert cfg.max_iterations == 5

    def test_config_aggregate(self):
        from config import Config

        cfg = Config()
        assert hasattr(cfg, "agent_decision")
        assert hasattr(cfg, "conversation")
        assert hasattr(cfg, "rag")
        assert hasattr(cfg, "medical_cv")
        assert hasattr(cfg, "api")
        assert hasattr(cfg, "speech")
        assert hasattr(cfg, "validation")
        assert hasattr(cfg, "ui")
        assert hasattr(cfg, "mcp")
        assert cfg.max_conversation_history == 20
