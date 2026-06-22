"""
Multi-provider LLM configuration for domestic (Chinese) and international models.

Usage:
    1. Set PROVIDER env var to one of the preset names below, OR
    2. Set OPENAI_BASE_URL + OPENAI_API_KEY + MODEL_NAME directly

Preset providers:
    xiaomi-mimo   - 小米 MiMo v2.5 Pro (via xiaomimimo.com)
    deepseek      - DeepSeek v4 Pro
    bybing        - bybing.cc OpenAI-compatible proxy
    openai        - OpenAI official
    qwen          - 阿里通义千问 (DashScope compatible mode)
    moonshot       - Kimi / Moonshot
    zhipu         - 智谱 GLM-4

You can also load from a mykey.py file (GA-style):
    providers.load_mykey("path/to/mykey.py")
"""

import os
import importlib.util
import logging

logger = logging.getLogger(__name__)

# ─── Preset Providers ────────────────────────────────────────────────────────
PRESETS = {
    "xiaomi-mimo": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "embedding_model": "text-embedding-3-large",  # fallback
        "description": "小米 MiMo v2.5 Pro",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-pro",
        "embedding_model": "text-embedding-v2",
        "description": "DeepSeek v4 Pro",
    },
    "bybing": {
        "base_url": "https://api.bybing.cc/v1",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "description": "bybing.cc OpenAI-compatible proxy",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "description": "OpenAI Official",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "embedding_model": "text-embedding-v3",
        "description": "阿里通义千问 (DashScope)",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-128k",
        "embedding_model": None,  # Moonshot doesn't have embeddings
        "description": "Kimi / Moonshot v1",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "embedding_model": "embedding-3",
        "description": "智谱 GLM-4",
    },
}


def load_mykey(filepath: str) -> dict:
    """Load provider config from a mykey.py file (GA-style).

    Looks for variables named like:
        native_oai_config_XXX = {'name': ..., 'apikey': ..., 'apibase': ..., 'model': ...}

    Returns dict of {name: {base_url, model, api_key}}.
    """
    if not os.path.exists(filepath):
        logger.warning(f"mykey.py not found at {filepath}")
        return {}

    spec = importlib.util.spec_from_file_location("mykey", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = {}
    for attr_name in dir(mod):
        if attr_name.startswith("native_oai_config_"):
            cfg = getattr(mod, attr_name)
            if isinstance(cfg, dict) and "apibase" in cfg:
                name = cfg.get("name", attr_name.replace("native_oai_config_", ""))
                configs[name] = {
                    "base_url": cfg["apibase"],
                    "model": cfg.get("model", ""),
                    "api_key": cfg.get("apikey", ""),
                }
                # Register as preset
                PRESETS[name] = {
                    "base_url": cfg["apibase"],
                    "model": cfg.get("model", ""),
                    "embedding_model": None,
                    "description": f"From mykey.py: {name}",
                }
    if configs:
        logger.info(f"Loaded {len(configs)} providers from {filepath}")
    return configs


def resolve_provider() -> dict:
    """Resolve the active provider config from environment variables.

    Priority:
        1. PROVIDER env var → lookup in PRESETS
        2. Direct env vars: OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME
        3. Fallback to 'bybing' preset (existing .env)

    Returns dict with keys: base_url, model, api_key, embedding_model, description
    """
    provider_name = os.getenv("PROVIDER", "").strip().lower()

    # Try preset lookup
    if provider_name and provider_name in PRESETS:
        preset = PRESETS[provider_name]
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key", "")
        result = {
            "base_url": preset["base_url"],
            "model": preset["model"],
            "api_key": api_key,
            "embedding_model": preset.get("embedding_model") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            "description": preset["description"],
        }
        logger.info(f"[Provider] Using preset '{provider_name}': {result['model']} @ {result['base_url']}")
        return result

    # Direct env vars (no preset)
    if os.getenv("OPENAI_BASE_URL"):
        result = {
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("MODEL_NAME") or os.getenv("model_name", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key", ""),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            "description": "Custom (env vars)",
        }
        logger.info(f"[Provider] Using custom env: {result['model']} @ {result['base_url']}")
        return result

    # Fallback: bybing (matches existing .env)
    result = {
        "base_url": "https://api.bybing.cc/v1",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key", ""),
        "embedding_model": "text-embedding-3-large",
        "description": "bybing.cc (fallback)",
    }
    logger.info(f"[Provider] Using fallback bybing: {result['model']}")
    return result


def list_providers() -> dict:
    """Return all available presets for UI/display."""
    return {k: v["description"] for k, v in PRESETS.items()}
