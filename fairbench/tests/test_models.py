"""Tests for fairbench.models — GGUF model catalog."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fairbench.models import ModelCatalog, ModelInfo, _parse_gguf_header


# --- Fixtures ---

def _create_mock_gguf(tmp_path: Path, name: str, arch: str = "llama") -> Path:
    """Create a minimal mock GGUF v3 file for testing."""
    gguf_path = tmp_path / f"{name}.gguf"

    # GGUF v3 format: magic + version + tensor_count + kv_count
    data = b"GGUF"  # magic
    data += struct.pack("<Q", 3)  # version (v3)
    data += struct.pack("<Q", 0)  # tensor count
    data += struct.pack("<Q", 2)  # KV count

    # KV pair 1: general.name (string)
    key1 = b"general.name"
    val1 = name.encode("utf-8")
    data += struct.pack("<Q", len(key1))  # key length (uint64 for v3)
    data += key1  # key
    data += struct.pack("<I", 10)  # value type: string (10)
    data += struct.pack("<Q", len(val1))  # string length (uint64)
    data += val1  # string value

    # KV pair 2: general.architecture (string)
    key2 = b"general.architecture"
    val2 = arch.encode("utf-8")
    data += struct.pack("<Q", len(key2))
    data += key2
    data += struct.pack("<I", 10)
    data += struct.pack("<Q", len(val2))
    data += val2

    gguf_path.write_bytes(data)
    return gguf_path


def _create_invalid_gguf(tmp_path: Path) -> Path:
    """Create a file with .gguf extension but invalid content."""
    gguf_path = tmp_path / "invalid.gguf"
    gguf_path.write_bytes(b"not a gguf file at all")
    return gguf_path


class TestModelInfo:
    def test_to_dict(self):
        """ModelInfo.to_dict serializes correctly."""
        info = ModelInfo(
            path=Path("/tmp/test.gguf"),
            name="test-model",
            architecture="llama",
            context_length=4096,
            quantization_version=2,
            file_size_bytes=4 * 1024**3,
            quantization="Q4_K_M",
        )
        d = info.to_dict()
        assert d["name"] == "test-model"
        assert d["architecture"] == "llama"
        assert d["context_length"] == 4096
        assert d["file_size_bytes"] == 4 * 1024**3

    def test_from_dict(self):
        """ModelInfo.from_dict deserializes correctly."""
        d = {
            "path": "/tmp/test.gguf",
            "name": "test-model",
            "architecture": "llama",
            "context_length": 4096,
            "quantization_version": 2,
            "file_size_bytes": 4 * 1024**3,
            "quantization": "Q4_K_M",
        }
        info = ModelInfo.from_dict(d)
        assert info.name == "test-model"
        assert info.path == Path("/tmp/test.gguf")

    def test_file_size_gb(self):
        """file_size_gb computes correctly."""
        info = ModelInfo(path=Path("/tmp/test.gguf"), file_size_bytes=8 * 1024**3)
        assert info.file_size_gb == 8.0


class TestGGUFHeaderParsing:
    def test_parse_valid_gguf(self, tmp_path):
        """_parse_gguf_header extracts metadata from valid GGUF."""
        gguf_path = _create_mock_gguf(tmp_path, "qwen2.5-7b", "qwen2")
        header = _parse_gguf_header(gguf_path)

        assert header["name"] == "qwen2.5-7b"
        assert header["architecture"] == "qwen2"
        assert header["file_size_bytes"] > 0

    def test_parse_invalid_gguf(self, tmp_path):
        """_parse_gguf_header returns defaults for invalid files."""
        gguf_path = _create_invalid_gguf(tmp_path)
        header = _parse_gguf_header(gguf_path)

        assert header["name"] == ""
        assert header["architecture"] == ""


class TestModelCatalog:
    def test_add_and_get_model(self, tmp_path):
        """Catalog.add_model and get_model work correctly."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        gguf_path = _create_mock_gguf(tmp_path, "qwen2.5-7b")

        info = catalog.add_model(gguf_path)
        assert info.name == "qwen2.5-7b"

        retrieved = catalog.get_model("qwen2.5-7b")
        assert retrieved is not None
        assert retrieved.name == "qwen2.5-7b"

    def test_remove_model(self, tmp_path):
        """Catalog.remove_model removes and persists."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        gguf_path = _create_mock_gguf(tmp_path, "llama3.1-8b")

        catalog.add_model(gguf_path)
        assert catalog.remove_model("llama3.1-8b") is True
        assert catalog.get_model("llama3.1-8b") is None

    def test_search(self, tmp_path):
        """Catalog.search finds models by pattern."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        _create_mock_gguf(tmp_path, "qwen2.5-7b-q4_k_m")
        _create_mock_gguf(tmp_path, "llama3.1-8b-q4_k_m")

        catalog.add_model(_create_mock_gguf(tmp_path, "qwen2.5-7b-q4_k_m"))
        catalog.add_model(_create_mock_gguf(tmp_path, "llama3.1-8b-q4_k_m"))

        results = catalog.search("qwen")
        assert len(results) == 1
        assert "qwen" in results[0].name.lower()

    def test_list_models(self, tmp_path):
        """Catalog.list_models returns sorted list."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        _create_mock_gguf(tmp_path, "alpha-model")
        _create_mock_gguf(tmp_path, "beta-model")

        catalog.add_model(_create_mock_gguf(tmp_path, "alpha-model"))
        catalog.add_model(_create_mock_gguf(tmp_path, "beta-model"))

        models = catalog.list_models()
        assert len(models) == 2
        assert models[0].name == "alpha-model"
        assert models[1].name == "beta-model"

    def test_persists_index(self, tmp_path):
        """Catalog saves and reloads index correctly."""
        index_path = tmp_path / "index.json"
        gguf_path = _create_mock_gguf(tmp_path, "persist-test")

        catalog1 = ModelCatalog(index_path=index_path)
        catalog1.add_model(gguf_path)

        # Reload from disk
        catalog2 = ModelCatalog(index_path=index_path)
        info = catalog2.get_model("persist-test")
        assert info is not None
        assert info.name == "persist-test"

    def test_get_model_by_path(self, tmp_path):
        """Catalog.get_model works with path lookup."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        gguf_path = _create_mock_gguf(tmp_path, "path-test")

        catalog.add_model(gguf_path)
        info = catalog.get_model(str(gguf_path))
        assert info is not None
        assert info.name == "path-test"

    def test_get_model_by_filename(self, tmp_path):
        """Catalog.get_model works with filename lookup."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        gguf_path = _create_mock_gguf(tmp_path, "file-test")

        catalog.add_model(gguf_path)
        info = catalog.get_model(gguf_path.name)
        assert info is not None
        assert info.name == "file-test"

    def test_search_case_insensitive(self, tmp_path):
        """Catalog.search is case-insensitive."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        catalog.add_model(_create_mock_gguf(tmp_path, "Qwen2.5-7B"))

        results_lower = catalog.search("qwen")
        results_upper = catalog.search("QWEN")
        assert len(results_lower) == 1
        assert len(results_upper) == 1

    def test_add_nonexistent_file(self, tmp_path):
        """Catalog.add_model raises for nonexistent file."""
        catalog = ModelCatalog(index_path=tmp_path / "index.json")
        with pytest.raises(FileNotFoundError):
            catalog.add_model(tmp_path / "nonexistent.gguf")
