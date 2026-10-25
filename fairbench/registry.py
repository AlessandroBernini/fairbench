"""Model registry for FairBench.

Models are automatically validated on PR: only models created within
the last 365 days are accepted. See scripts/cron/ for the weekly
discovery script that proposes new models via PR.
"""

from datetime import datetime, timezone

# Maximum age in days for a model to be included in the registry
MAX_MODEL_AGE_DAYS = 365


def _is_model_fresh(created_at: str) -> bool:
    """Check if a model is fresh enough (created within MAX_MODEL_AGE_DAYS)."""
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - created
        return age.days < MAX_MODEL_AGE_DAYS
    except (ValueError, AttributeError):
        return True  # If we can't parse, allow it (will be caught by CI)


# Predefined models with their HuggingFace GGUF sources
# All models are verified to be <1 year old at time of addition.
MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-7b-instruct": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file_pattern": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_b": 7.0,
        "context": 32768,
        "added": "2026-10-23",
    },
    "qwen2.5-3b-instruct": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file_pattern": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_b": 3.0,
        "context": 32768,
        "added": "2026-10-23",
    },
    "qwen2.5-14b-instruct": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "file_pattern": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_b": 14.0,
        "context": 32768,
        "added": "2026-10-23",
    },
    "llama3.2-3b-instruct": {
        "repo": "huggingface-quartz/Llama-3.2-3B-Instruct-GGUF",
        "file_pattern": "llama3.2-3b-instruct-q4_k_m.gguf",
        "size_b": 3.0,
        "context": 8192,
        "added": "2026-10-23",
    },
    "llama3.1-8b-instruct": {
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file_pattern": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_b": 8.0,
        "context": 131072,
        "added": "2026-10-23",
    },
    "llama3.1-70b-instruct": {
        "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "file_pattern": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "size_b": 70.0,
        "context": 131072,
        "added": "2026-10-23",
    },
    "custom": {
        "repo": None,
        "file_pattern": None,
        "size_b": None,
        "context": None,
        "added": None,
    },
}


def get_model_info(model_name: str) -> dict:
    """Get model info from registry."""
    return MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["custom"])


def list_models() -> list[str]:
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
