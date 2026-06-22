"""Tests for Phase 8: Startup Configuration Validator."""
import os
import pytest
from unittest.mock import patch

from startup_validator import (
    ConfigValidationError,
    ValidationResult,
    _check_required_env,
    _check_model_config,
    _check_qdrant_config,
    _check_directory_structure,
    validate_startup_config,
)


class TestValidationResult:
    """Test ValidationResult data structure."""

    def test_default_values(self):
        result = ValidationResult()
        assert result.errors == []
        assert result.warnings == []
        assert result.valid is True

    def test_add_error(self):
        result = ValidationResult()
        result.add_error("test error")
        assert result.valid is False
        assert "test error" in result.errors

    def test_add_warning(self):
        result = ValidationResult()
        result.add_warning("test warning")
        assert result.valid is True  # warnings don't invalidate
        assert "test warning" in result.warnings


class TestCheckRequiredEnv:
    """Test required env var checks."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234"}, clear=False)
    def test_valid_key(self):
        result = ValidationResult()
        _check_required_env(result)
        assert result.valid is True

    @patch.dict(os.environ, {"openai_api_key": "sk-test123456789012345678901234"}, clear=False)
    def test_alt_env_name(self):
        os.environ.pop("OPENAI_API_KEY", None)
        result = ValidationResult()
        _check_required_env(result)
        assert result.valid is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_empty_key(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("openai_api_key", None)
        os.environ.pop("embedding_openai_api_key", None)
        result = ValidationResult()
        _check_required_env(result)
        assert result.valid is False
        assert any("API key" in e for e in result.errors)


class TestCheckModelConfig:
    """Test model configuration check."""

    @patch.dict(os.environ, {"model_name": "gpt-4o-mini"}, clear=False)
    def test_model_set(self):
        result = ValidationResult()
        _check_model_config(result)
        assert len(result.warnings) == 0

    @patch.dict(os.environ, {"model_name": ""}, clear=False)
    def test_model_missing(self):
        os.environ.pop("model_name", None)
        result = ValidationResult()
        _check_model_config(result)
        assert len(result.warnings) == 1
        assert "defaulting" in result.warnings[0].lower()


class TestCheckQdrantConfig:
    """Test Qdrant configuration check."""

    @patch.dict(os.environ, {"QDRANT_URL": "http://localhost:6333"}, clear=False)
    def test_remote_without_key_warns(self):
        os.environ.pop("QDRANT_API_KEY", None)
        result = ValidationResult()
        _check_qdrant_config(result)
        assert any("QDRANT_API_KEY" in w for w in result.warnings)

    @patch.dict(os.environ, {"QDRANT_URL": "http://localhost:6333", "QDRANT_API_KEY": "test"}, clear=False)
    def test_remote_with_key_ok(self):
        result = ValidationResult()
        _check_qdrant_config(result)
        assert not any("QDRANT_API_KEY" in w for w in result.warnings)


class TestCheckDirectoryStructure:
    """Test directory structure checks."""

    def test_existing_dirs_pass(self):
        result = ValidationResult()
        _check_directory_structure(result)
        # agents, api, middleware should exist
        assert not any("agents" in w for w in result.warnings)


class TestValidateStartupConfig:
    """Test the main validation entry point."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234",
        "model_name": "gpt-4o-mini",
    }, clear=False)
    def test_valid_config_returns_result(self):
        result = validate_startup_config()
        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_missing_key_raises(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("openai_api_key", None)
        os.environ.pop("embedding_openai_api_key", None)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_startup_config()
        assert len(exc_info.value.errors) >= 1

    def test_config_validation_error_message(self):
        err = ConfigValidationError(["error1", "error2"])
        assert "error1" in str(err)
        assert "error2" in str(err)
        assert err.errors == ["error1", "error2"]
