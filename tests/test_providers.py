"""
Unit tests for providers.py — Multi-provider LLM configuration.
Run: python -m pytest tests/test_providers.py -v
"""

import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import PRESETS, list_providers, load_mykey, resolve_provider

# ── PRESETS ──────────────────────────────────────────────────────────────────


class TestPresets:
    """Test that all expected presets exist with required keys."""

    def test_all_preset_names(self):
        expected = {"xiaomi-mimo", "deepseek", "bybing", "openai", "qwen", "moonshot", "zhipu"}
        assert expected == set(PRESETS.keys())

    def test_preset_has_required_keys(self):
        for name, preset in PRESETS.items():
            assert "base_url" in preset, f"{name} missing base_url"
            assert "model" in preset, f"{name} missing model"
            assert "description" in preset, f"{name} missing description"

    def test_xiaomi_mimo_values(self):
        p = PRESETS["xiaomi-mimo"]
        assert "xiaomimimo.com" in p["base_url"]
        assert p["model"] == "mimo-v2.5-pro"

    def test_deepseek_values(self):
        p = PRESETS["deepseek"]
        assert "deepseek.com" in p["base_url"]
        assert p["model"] == "deepseek-v4-pro"

    def test_moonshot_no_embedding(self):
        p = PRESETS["moonshot"]
        assert p["embedding_model"] is None

    def test_all_base_urls_have_version_path(self):
        """All presets should use versioned endpoint (v1 or v4)."""
        for name, preset in PRESETS.items():
            url = preset["base_url"]
            has_version = "/v1" in url or "/v4" in url
            assert has_version, f"{name} base_url missing version path: {url}"


# ── list_providers ───────────────────────────────────────────────────────────


class TestListProviders:
    """Test list_providers() returns display-friendly dict."""

    def test_returns_all_presets(self):
        result = list_providers()
        assert len(result) == len(PRESETS)

    def test_values_are_descriptions(self):
        result = list_providers()
        for _name, desc in result.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_known_providers_present(self):
        result = list_providers()
        assert "xiaomi-mimo" in result
        assert "deepseek" in result
        assert "openai" in result


# ── resolve_provider ─────────────────────────────────────────────────────────


class TestResolveProvider:
    """Test resolve_provider() with different env var configurations."""

    def test_preset_by_env(self):
        """PROVIDER env var should select preset."""
        with patch.dict(
            os.environ,
            {
                "PROVIDER": "deepseek",
                "OPENAI_API_KEY": "test-key-123",
            },
        ):
            result = resolve_provider()
            assert result["base_url"] == "https://api.deepseek.com/v1"
            assert result["model"] == "deepseek-v4-pro"
            assert result["api_key"] == "test-key-123"
            assert "DeepSeek" in result["description"]

    def test_xiaomi_mimo_preset(self):
        with patch.dict(
            os.environ,
            {
                "PROVIDER": "xiaomi-mimo",
                "OPENAI_API_KEY": "sk-test",
            },
        ):
            result = resolve_provider()
            assert "xiaomimimo.com" in result["base_url"]
            assert result["model"] == "mimo-v2.5-pro"

    def test_custom_env_vars(self):
        """OPENAI_BASE_URL + MODEL_NAME should override presets."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://custom.api.com/v1",
                "MODEL_NAME": "custom-model",
                "OPENAI_API_KEY": "custom-key",
            },
            clear=False,
        ):
            # Remove PROVIDER to avoid preset lookup
            os.environ.pop("PROVIDER", None)
            result = resolve_provider()
            assert result["base_url"] == "https://custom.api.com/v1"
            assert result["model"] == "custom-model"
            assert result["api_key"] == "custom-key"

    def test_fallback_to_bybing(self):
        """No PROVIDER + no OPENAI_BASE_URL → fallback to bybing."""
        env = {
            "OPENAI_API_KEY": "fallback-key",
        }
        # Clear all relevant vars
        for key in ["PROVIDER", "OPENAI_BASE_URL", "MODEL_NAME"]:
            env.pop(key, None)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("PROVIDER", None)
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("MODEL_NAME", None)
            result = resolve_provider()
            assert "bybing" in result["base_url"].lower() or "bybing" in result["description"].lower()

    def test_unknown_preset_falls_through(self):
        """Unknown PROVIDER name should fall through to custom or fallback."""
        with patch.dict(
            os.environ,
            {
                "PROVIDER": "nonexistent-provider",
                "OPENAI_BASE_URL": "https://fallback.api.com/v1",
                "OPENAI_API_KEY": "key",
            },
            clear=False,
        ):
            result = resolve_provider()
            # Should use OPENAI_BASE_URL
            assert result["base_url"] == "https://fallback.api.com/v1"

    def test_preset_embedding_model(self):
        """Preset embedding_model should be carried through."""
        with patch.dict(
            os.environ,
            {
                "PROVIDER": "qwen",
                "OPENAI_API_KEY": "key",
            },
        ):
            result = resolve_provider()
            assert result["embedding_model"] == "text-embedding-v3"

    def test_preset_none_embedding_fallback(self):
        """If preset embedding_model is None, should fall back to env or default."""
        with patch.dict(
            os.environ,
            {
                "PROVIDER": "moonshot",
                "OPENAI_API_KEY": "key",
            },
        ):
            result = resolve_provider()
            # Should fall back to default
            assert result["embedding_model"] == "text-embedding-3-large"


# ── load_mykey ───────────────────────────────────────────────────────────────


class TestLoadMykey:
    """Test load_mykey() GA-style config loading."""

    def test_load_valid_mykey(self):
        """Should parse native_oai_config_XXX variables."""
        content = """native_oai_config_test = {
    "name": "test-provider",
    "apikey": "sk-test-key",
    "apibase": "https://test.api.com/v1",
    "model": "test-model-v1",
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            configs = load_mykey(path)
            assert "test-provider" in configs
            assert configs["test-provider"]["base_url"] == "https://test.api.com/v1"
            assert configs["test-provider"]["model"] == "test-model-v1"
            assert configs["test-provider"]["api_key"] == "sk-test-key"
            # Should also register as preset
            assert "test-provider" in PRESETS
            assert PRESETS["test-provider"]["description"] == "From mykey.py: test-provider"
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        """Should return empty dict for missing file."""
        result = load_mykey("/nonexistent/path/mykey.py")
        assert result == {}

    def test_load_mykey_no_matching_vars(self):
        """File without native_oai_config_XXX should return empty."""
        content = 'some_other_var = {"key": "value"}\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            configs = load_mykey(path)
            assert configs == {}
        finally:
            os.unlink(path)

    def test_load_mykey_multiple_configs(self):
        """Should parse multiple native_oai_config_XXX variables."""
        content = """
native_oai_config_alpha = {
    "name": "alpha",
    "apikey": "sk-alpha",
    "apibase": "https://alpha.api.com/v1",
    "model": "alpha-1",
}
native_oai_config_beta = {
    "name": "beta",
    "apikey": "sk-beta",
    "apibase": "https://beta.api.com/v1",
    "model": "beta-2",
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            configs = load_mykey(path)
            assert len(configs) == 2
            assert "alpha" in configs
            assert "beta" in configs
        finally:
            os.unlink(path)

    def test_load_mykey_uses_attr_name_as_fallback(self):
        """If config dict has no 'name' key, should derive from attr name."""
        content = """native_oai_config_mynodef = {
    "apikey": "sk-xxx",
    "apibase": "https://xxx.api.com/v1",
    "model": "xxx-v1",
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            configs = load_mykey(path)
            assert "mynodef" in configs
        finally:
            os.unlink(path)
