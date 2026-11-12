"""Tests for fairbench.hardware_profile — hardware profiling and caching."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from fairbench.hardware_profile import (
    HardwareSnapshot,
    clear_hardware_cache,
    get_cached_hardware_snapshot,
    get_hardware_profile,
    get_hw_profile_info,
    save_hardware_snapshot,
)


# --- Fixtures ---
def _create_mock_profile(tmp_path: Path) -> Path:
    """Create a mock hardware profile cache file."""
    profile = HardwareSnapshot(
        cpu_model="Test CPU",
        cpu_cores=4,
        cpu_threads=8,
        total_ram_gb=16.0,
        gpu_available=True,
        gpu_type="nvidia",
        gpu_vram_gb=8.0,
        os_platform="linux",
        os_version="6.1.0",
        kernel_version="6.1.0-rt",
    )
    cache_file = tmp_path / "hardware_profile.json"
    with open(cache_file, "w") as f:
        json.dump(profile.to_dict(), f)
    return cache_file


class TestHardwareSnapshot:
    def test_to_dict(self):
        """HardwareSnapshot.to_dict serializes correctly."""
        snapshot = HardwareSnapshot(
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_threads=8,
            total_ram_gb=16.0,
            gpu_available=True,
            gpu_type="nvidia",
            gpu_vram_gb=8.0,
            os_platform="linux",
            os_version="6.1.0",
            kernel_version="6.1.0-rt",
        )
        d = snapshot.to_dict()
        assert "profile_hash" not in d
        assert d["cpu_model"] == "Test CPU"
        assert d["gpu_available"] is True

    def test_from_dict(self):
        """HardwareSnapshot.from_dict deserializes correctly."""
        d = {
            "cpu_model": "Test CPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "total_ram_gb": 16.0,
            "gpu_available": True,
            "gpu_type": "nvidia",
            "gpu_vram_gb": 8.0,
            "os_platform": "linux",
            "os_version": "6.1.0",
            "kernel_version": "6.1.0-rt",
        }
        snapshot = HardwareSnapshot.from_dict(d)
        assert snapshot.cpu_model == "Test CPU"
        assert snapshot.profile_hash != ""

    def test_profile_hash_consistent(self):
        """Same input produces same profile_hash."""
        d1 = {
            "cpu_model": "Test CPU",
            "cpu_cores": 4,
            "cpu_threads": 8,
            "total_ram_gb": 16.0,
            "gpu_available": True,
            "gpu_type": "nvidia",
            "gpu_vram_gb": 8.0,
            "os_platform": "linux",
            "os_version": "6.1.0",
            "kernel_version": "6.1.0-rt",
        }
        s1 = HardwareSnapshot.from_dict(d1)
        s2 = HardwareSnapshot.from_dict(d1)
        assert s1.profile_hash == s2.profile_hash


class TestHardwareCache:
    def test_save_and_load_cache(self, tmp_path, monkeypatch):
        """save_hardware_snapshot writes file that can be read back."""
        cache_file = tmp_path / "hardware_profile.json"
        profile = HardwareSnapshot(
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_threads=8,
            total_ram_gb=16.0,
            gpu_available=True,
            gpu_type="nvidia",
            gpu_vram_gb=8.0,
            os_platform="linux",
            os_version="6.1.0",
            kernel_version="6.1.0-rt",
        )

        import fairbench.hardware_profile as hp

        monkeypatch.setattr(hp, "_CACHE_FILE", cache_file)
        saved = hp.save_hardware_snapshot(profile)
        assert saved == cache_file
        assert cache_file.exists()

        with open(cache_file, "r") as f:
            data = json.load(f)
        assert data["cpu_model"] == "Test CPU"

    def test_clear_cache(self, tmp_path, monkeypatch):
        """clear_hardware_cache removes the cache file."""
        cache_file = tmp_path / "hardware_profile.json"
        cache_file.write_text("{}")

        import fairbench.hardware_profile as hp

        monkeypatch.setattr(hp, "_CACHE_FILE", cache_file)
        result = hp.clear_hardware_cache()
        assert result is True
        assert not cache_file.exists()

    def test_clear_missing_cache(self):
        """clear_hardware_cache returns False when no cache exists."""
        import fairbench.hardware_profile as hp

        fake_file = Path("/tmp/nonexistent_hardware_profile.json")
        monkeypatch = None  # type: ignore[unused-ignore] # noqa: F841

        with patch.object(hp, "_CACHE_FILE", fake_file):
            result = hp.clear_hardware_cache()
            assert result is False


class TestGetHwProfileInfo:
    def test_returns_expected_keys(self):
        """get_hw_profile_info returns all expected keys."""
        info = get_hw_profile_info()
        expected_keys = {
            "profile_hash",
            "cpu_model",
            "cpu_cores",
            "cpu_threads",
            "total_ram_gb",
            "gpu_available",
            "gpu_type",
            "gpu_vram_gb",
            "os_platform",
            "os_version",
            "kernel_version",
            "cached",
        }
        assert expected_keys.issubset(set(info.keys()))

    def test_profile_hash_not_empty(self):
        """get_hw_profile_info returns non-empty profile_hash."""
        info = get_hw_profile_info()
        assert info["profile_hash"] != ""

    def test_gpu_available_is_bool(self):
        """gpu_available is a boolean."""
        info = get_hw_profile_info()
        assert isinstance(info["gpu_available"], bool)
