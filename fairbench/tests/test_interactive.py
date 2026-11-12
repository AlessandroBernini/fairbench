"""Tests for fairbench.interactive — interactive benchmark mode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fairbench.interactive import _configure_parameters, _review_and_confirm, _select_category, _select_model
from fairbench.models import ModelCatalog


# --- Fixtures ---
def _create_mock_model(tmp_path: Path, name: str, arch: str = "llama", size_gb: float = 4.0) -> Path:
    """Create a minimal mock GGUF file for testing."""
    import struct

    gguf_path = tmp_path / f"{name}.gguf"
    data = b"GGUF"
    data += struct.pack("<Q", 3)  # version
    data += struct.pack("<Q", 0)  # tensor count
    data += struct.pack("<Q", 2)  # KV count

    key1 = b"general.name"
    val1 = name.encode("utf-8")
    data += struct.pack("<Q", len(key1))
    data += key1
    data += struct.pack("<I", 10)
    data += struct.pack("<Q", len(val1))
    data += val1

    key2 = b"general.architecture"
    val2 = arch.encode("utf-8")
    data += struct.pack("<Q", len(key2))
    data += key2
    data += struct.pack("<I", 10)
    data += struct.pack("<Q", len(val2))
    data += val2

    gguf_path.write_bytes(data)
    return gguf_path


class TestSelectModel:
    def test_select_from_catalog(self, tmp_path):
        """_select_model picks catalog model when available."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        gguf_path = _create_mock_model(tmp_path, "test-model-1", "llama", 4.0)
        catalog.add_model(gguf_path)

        with patch("click.prompt", return_value=1):
            result = _select_model(catalog)
            assert result == str(gguf_path)

    def test_fallback_to_path_input(self, tmp_path):
        """_select_model accepts path input when catalog is empty."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")

        def prompt_side_effect(*args, **kwargs):
            return "/custom/path/model.gguf"

        with patch("click.prompt", side_effect=prompt_side_effect):
            result = _select_model(catalog)
            assert result == "/custom/path/model.gguf"


class TestSelectCategory:
    def test_select_quality(self):
        """_select_category returns 'quality' when selected."""
        with patch("click.prompt", return_value=1):
            result = _select_category()
            assert result == "quality"

    def test_select_all(self):
        """_select_category returns 'all' when selected."""
        with patch("click.prompt", return_value=5):
            result = _select_category()
            assert result == "all"

    def test_invalid_category_defaults_to_all(self):
        """_select_category returns 'all' for invalid selection."""
        with patch("click.prompt", return_value=10):
            result = _select_category()
            assert result == "all"


class TestConfigureParameters:
    def test_default_parameters(self):
        """_configure_parameters returns expected defaults."""
        with patch("click.prompt", side_effect=[4096, -1, 4, ""]):
            params = _configure_parameters("/tmp/model.gguf")
            assert params["n_ctx"] == 4096
            assert params["n_gpu_layers"] == -1
            assert params["n_threads"] == 4
            assert params["quantization"] is None

    def test_custom_parameters(self):
        """_configure_parameters accepts custom values."""
        with patch("click.prompt", side_effect=[8192, 32, 8, "Q5_K_M"]):
            params = _configure_parameters("/tmp/model.gguf")
            assert params["n_ctx"] == 8192
            assert params["n_gpu_layers"] == 32
            assert params["n_threads"] == 8
            assert params["quantization"] == "Q5_K_M"


class TestReviewAndConfirm:
    def test_confirm_true(self):
        """_review_and_confirm returns True when confirmed."""
        with patch("click.confirm", return_value=True):
            result = _review_and_confirm("/tmp/model.gguf", "quality", {"n_ctx": 4096, "n_gpu_layers": -1, "n_threads": 4, "quantization": None})
            assert result is True

    def test_confirm_false(self):
        """_review_and_confirm returns False when cancelled."""
        with patch("click.confirm", return_value=False):
            result = _review_and_confirm("/tmp/model.gguf", "quality", {"n_ctx": 4096, "n_gpu_layers": -1, "n_threads": 4, "quantization": None})
            assert result is False
