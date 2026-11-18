"""Tests for fairbench.data.DataManager (F6 coverage)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from fairbench.data import DataManager


@pytest.fixture
def dm(tmp_path: Path) -> DataManager:
    return DataManager(results_dir=str(tmp_path / "results"))


class TestSaveLoadResult:
    def test_save_and_load_roundtrip(self, dm: DataManager) -> None:
        payload = {"model": "qwen", "score": 42}
        path = dm.save_result("run_1", payload)
        assert path.exists()
        assert dm.load_result("run_1") == payload

    def test_load_missing_returns_none(self, dm: DataManager) -> None:
        assert dm.load_result("nope") is None

    def test_save_creates_category_dir(self, dm: DataManager) -> None:
        path = dm.save_result("r2", {"a": 1}, category="custom")
        assert path.parent.name == "custom"
        assert dm.load_result("r2", category="custom") == {"a": 1}

    def test_list_results_empty(self, dm: DataManager) -> None:
        assert dm.list_results() == []

    def test_list_results_after_saves(self, dm: DataManager) -> None:
        dm.save_result("run_a", {})
        dm.save_result("run_b", {})
        assert sorted(dm.list_results()) == ["run_a", "run_b"]


class TestSwebenchInstances:
    def test_save_and_load_instance(self, dm: DataManager) -> None:
        data = {"instance_id": "repo__pr1", "problem_statement": "fix"}
        path = dm.save_swebench_instance("repo__pr1", data)
        assert path.exists()
        assert dm.load_swebench_instance("repo__pr1") == data

    def test_load_missing_instance_returns_none(self, dm: DataManager) -> None:
        assert dm.load_swebench_instance("missing") is None


class TestModelHashing:
    def test_get_model_hash_matches_sha256(self, dm: DataManager, tmp_path: Path) -> None:
        model = tmp_path / "model.gguf"
        content = b"fake gguf bytes" * 1000
        model.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert dm.get_model_hash(str(model)) == expected

    def test_cache_key_falls_back_when_metadata_parse_fails(self, dm: DataManager, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with patch("fairbench.data._parse_gguf_header", side_effect=OSError("boom")):
            file_hash, meta_hash = dm.get_model_cache_key(str(model), "speed")
        assert len(file_hash) == 64
        assert meta_hash == ""

    def test_cache_key_missing_file_raises(self, dm: DataManager, tmp_path: Path) -> None:
        # get_model_hash runs before the try block → FileNotFoundError propagates.
        missing = tmp_path / "does_not_exist.gguf"
        with pytest.raises(FileNotFoundError):
            dm.get_model_cache_key(str(missing), "speed")

    def test_cache_key_non_gguf_uses_default_metadata(self, dm: DataManager, tmp_path: Path) -> None:
        junk = tmp_path / "junk.bin"
        junk.write_bytes(b"not a gguf")
        file_hash, meta_hash = dm.get_model_cache_key(str(junk), "speed")
        assert len(file_hash) == 64
        # _parse_gguf_header returns default keys (no raise); hash excludes file_size_bytes
        expected_str = str(
            sorted(["name", "architecture", "context_length", "quantization_version", "quantization"])
        )
        assert meta_hash == hashlib.md5(expected_str.encode()).hexdigest()

    def test_cache_key_uses_gguf_metadata(self, dm: DataManager, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        fake_meta = {"general.architecture": "qwen2", "file_size_bytes": 1}
        with patch("fairbench.data._parse_gguf_header", return_value=fake_meta):
            file_hash, meta_hash = dm.get_model_cache_key(str(model), "speed")
        assert len(file_hash) == 64
        # metadata hash must exclude volatile file_size_bytes
        expected_str = str(sorted(["general.architecture"]))
        assert meta_hash == hashlib.md5(expected_str.encode()).hexdigest()


class TestCache:
    def test_get_cached_result_miss(self, dm: DataManager) -> None:
        assert dm.get_cached_result("h" * 64, "speed") is None

    def test_save_and_get_cached_result(self, dm: DataManager) -> None:
        dm.save_cached_result("hash1", "speed", {"ttft": 12.5})
        assert dm.get_cached_result("hash1", "speed") == {"ttft": 12.5}

    def test_ttl_writes_meta_file(self, dm: DataManager) -> None:
        dm.save_cached_result("hash2", "quality", [1, 2], ttl_days=7)
        meta_path = dm.cache_dir / "hash2_quality.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["ttl_days"] == 7
        assert abs(meta["cached_at"] - time.time()) < 60

    def test_is_cache_valid_without_meta(self, dm: DataManager) -> None:
        dm.save_cached_result("h3", "speed", {})
        assert dm.is_cache_valid("h3", "speed") is True

    def test_is_cache_valid_fresh_ttl(self, dm: DataManager) -> None:
        dm.save_cached_result("h4", "speed", {}, ttl_days=7)
        assert dm.is_cache_valid("h4", "speed") is True

    def test_is_cache_invalid_expired_ttl(self, dm: DataManager) -> None:
        dm.save_cached_result("h5", "speed", {}, ttl_days=1)
        meta_path = dm.cache_dir / "h5_speed.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["cached_at"] = time.time() - 2 * 86400  # 2 days old
        meta_path.write_text(json.dumps(meta))
        assert dm.is_cache_valid("h5", "speed") is False

    def test_is_cache_valid_corrupt_meta(self, dm: DataManager) -> None:
        dm.save_cached_result("h6", "speed", {}, ttl_days=1)
        (dm.cache_dir / "h6_speed.meta.json").write_text("{not json")
        assert dm.is_cache_valid("h6", "speed") is True

    def test_is_cache_missing_entry_true(self, dm: DataManager) -> None:
        # No meta file at all → treated as valid (no TTL semantics)
        assert dm.is_cache_valid("absent", "speed") is True

    def test_clear_cache_returns_count(self, dm: DataManager) -> None:
        dm.save_cached_result("a", "speed", {}, ttl_days=1)
        dm.save_cached_result("b", "quality", {})
        removed = dm.clear_cache()
        assert removed == 3  # 2 data + 1 meta json
        assert list(dm.cache_dir.glob("*.json")) == []
