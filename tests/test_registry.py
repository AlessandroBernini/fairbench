"""Tests for fairbench.registry module."""

from fairbench.registry import get_model_info, list_models


class TestRegistry:
    def test_list_models(self):
        """test_list_models: list_models returns list of model names."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "qwen2.5-7b-instruct" in models
        assert "custom" in models

    def test_get_model_info(self):
        """test_get_model_info: get_model_info returns correct info for known model."""
        info = get_model_info("qwen2.5-7b-instruct")
        assert info["repo"] == "Qwen/Qwen2.5-7B-Instruct-GGUF"
        assert info["size_b"] == 7.0
        assert info["context"] == 32768
        assert "file_pattern" in info

    def test_get_model_info_custom(self):
        """test_get_model_info_custom: get_model_info returns custom for unknown."""
        info = get_model_info("unknown-model")
        assert info["repo"] is None
        assert info["file_pattern"] is None

    def test_get_model_info_custom_key(self):
        """test_get_model_info_custom: get_model_info returns custom for 'custom' key."""
        info = get_model_info("custom")
        assert info["repo"] is None
        assert info["file_pattern"] is None

    def test_registry_has_qwen_models(self):
        """test_registry_has_qwen_models: registry includes Qwen models."""
        models = list_models()
        qwen_models = [m for m in models if "qwen" in m]
        assert len(qwen_models) >= 3  # 7b, 3b, 14b

    def test_registry_has_llama_models(self):
        """test_registry_has_llama_models: registry includes Llama models."""
        models = list_models()
        llama_models = [m for m in models if "llama" in m]
        assert len(llama_models) >= 2  # 3b-instruct, 8b-instruct, 70b-instruct

    def test_all_models_have_required_fields(self):
        """test_all_models_have_required_fields: every model has repo, file_pattern, size_b, context."""
        required = ["repo", "file_pattern", "size_b", "context", "added"]
        for name in list_models():
            info = get_model_info(name)
            for field in required:
                assert field in info, f"Missing field '{field}' for model '{name}'"
