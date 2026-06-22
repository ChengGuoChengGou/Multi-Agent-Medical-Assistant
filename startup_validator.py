"""
Startup configuration validator.

Validates required environment variables and configuration at application
startup, failing fast with clear error messages instead of crashing mid-request.

Usage:
    from startup_validator import validate_startup_config
    validate_startup_config()  # raises ConfigValidationError if invalid
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when startup configuration is invalid."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def _check_required_env(result: ValidationResult):
    """Check required environment variables."""
    # API key: try multiple env var names
    api_key = (
        os.getenv("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("embedding_openai_api_key")
    )
    if not api_key:
        result.add_error(
            "No LLM API key found. Set OPENAI_API_KEY or openai_api_key in .env"
        )


def _check_model_config(result: ValidationResult):
    """Check model configuration."""
    model_name = os.getenv("model_name")
    if not model_name:
        result.add_warning(
            "model_name not set, defaulting to gpt-4o-mini"
        )


def _check_provider_system(result: ValidationResult):
    """Check if provider system can be loaded."""
    try:
        from providers import resolve_provider
        provider = resolve_provider()
        if not provider.get("api_key"):
            result.add_error(f"Provider '{provider.get('name')}' returned no API key")
        if not provider.get("base_url"):
            result.add_warning(f"Provider '{provider.get('name')}' returned no base_url")
    except ImportError:
        result.add_warning("providers.py not found, using env vars directly")
    except Exception as e:
        result.add_error(f"Provider system failed to initialize: {e}")


def _check_qdrant_config(result: ValidationResult):
    """Check Qdrant vector DB configuration."""
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        # Remote Qdrant: needs API key
        if not os.getenv("QDRANT_API_KEY"):
            result.add_warning(
                "QDRANT_URL set but QDRANT_API_KEY missing (may fail for remote instances)"
            )
    else:
        # Local Qdrant: check data directory exists or can be created
        qdrant_path = os.path.join(".", "data", "qdrant_db")
        data_dir = os.path.join(".", "data")
        if not os.path.exists(data_dir):
            result.add_warning(
                f"Data directory '{data_dir}' not found. Qdrant local storage may fail."
            )


def _check_mcp_config(result: ValidationResult):
    """Check MCP server availability (non-fatal)."""
    mcp_commands = {
        "biomcp": ("uv", ["run", "biomcp", "stdio"]),
        "autoicd-mcp": ("uv", ["run", "python", "server.py"]),
        "healthcare-mcp": ("npx", ["-y", "@biantylabs/healthcare-mcp-public", "stdio"]),
    }
    import shutil
    for name, (cmd, _) in mcp_commands.items():
        if not shutil.which(cmd):
            result.add_warning(
                f"MCP server '{name}' requires '{cmd}' not found in PATH"
            )


def _check_directory_structure(result: ValidationResult):
    """Check required directories exist."""
    required_dirs = [
        ("agents", "Agent modules directory"),
        ("api", "API routes directory"),
        ("middleware", "Middleware directory"),
    ]
    for dirname, desc in required_dirs:
        if not os.path.isdir(dirname):
            result.add_warning(f"Directory '{dirname}' not found ({desc})")


def validate_startup_config() -> ValidationResult:
    """Run all startup validation checks.

    Returns:
        ValidationResult with errors and warnings.

    Raises:
        ConfigValidationError if any errors are found.
    """
    result = ValidationResult()

    _check_required_env(result)
    _check_model_config(result)
    _check_provider_system(result)
    _check_qdrant_config(result)
    _check_mcp_config(result)
    _check_directory_structure(result)

    # Log results
    if result.warnings:
        for w in result.warnings:
            logger.warning(f"[StartupValidator] {w}")

    if result.errors:
        for e in result.errors:
            logger.error(f"[StartupValidator] {e}")
        raise ConfigValidationError(result.errors)

    logger.info("[StartupValidator] All configuration checks passed")
    return result
