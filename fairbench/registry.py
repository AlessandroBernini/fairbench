"""Model registry for FairBench."""

from typing import Optional


# Predefined models with their HuggingFace GGUF sources
MODEL_REGISTRY: dict[str, dict] = {
    "qwen2.5-7b-instruct": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file_pattern": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_b": 7.0,
        "context": 32768,
    },
    "qwen2.5-3b-instruct": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file_pattern": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_b": 3.0,
        "context": 32768,
    },
    "llama3.2-3b-instruct": {
        "repo": "huggingface-quartz/Llama-3.2-3B-Instruct-GGUF",
        "file_pattern": "llama3.2-3b-instruct-q4_k_m.gguf",
        "size_b": 3.0,
        "context": 8192,
    },
    "phi3.5-mini-instruct": {
        "repo": "microsoft/Phi-3.5-mini-instruct-gguf",
        "file_pattern": "phi3.5-mini-instruct-q4.gguf",
        "size_b": 3.8,
        "context": 128000,
    },
    "gemma2-9b-instruct": {
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "file_pattern": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_b": 9.0,
        "context": 8192,
    },
    "mistral-7b-instruct": {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "file_pattern": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_b": 7.0,
        "context": 32768,
    },
    "llama3.1-8b-instruct": {
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file_pattern": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_b": 8.0,
        "context": 131072,
    },
    "qwen2.5-14b-instruct": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "file_pattern": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_b": 14.0,
        "context": 32768,
    },
    "llama3.1-70b-instruct": {
        "repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "file_pattern": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "size_b": 70.0,
        "context": 131072,
    },
    "custom": {
        "repo": None,
        "file_pattern": None,
        "size_b": None,
        "context": None,
    },
}


def get_model_info(model_name: str) -> dict:
    """Get model info from registry."""
    return MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["custom"])


def list_models() -> list[str]:
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
