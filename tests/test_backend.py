"""Tests for fairbench.backend module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fairbench.backend import detect_hardware, benchmark_memory


class TestDetectHardware:
    def test_returns_dict(self):
        """detect_hardware returns a dict with required keys."""
        hardware = detect_hardware()
        assert isinstance(hardware, dict)
        assert "cpu" in hardware
        assert "ram_gb" in hardware
        assert "gpu" in hardware
        assert "os" in hardware

    def test_cpu_info(self):
        """detect_hardware returns CPU info."""
        hardware = detect_hardware()
        assert "model" in hardware["cpu"]
        assert "cores" in hardware["cpu"]
        assert "threads" in hardware["cpu"]
        assert isinstance(hardware["cpu"]["cores"], int)
        assert isinstance(hardware["cpu"]["threads"], int)

    def test_ram_gb_positive(self):
        """detect_hardware returns positive RAM value."""
        hardware = detect_hardware()
        assert hardware["ram_gb"] > 0
        assert isinstance(hardware["ram_gb"], float)

    def test_gpu_type(self):
        """detect_hardware returns GPU type."""
        hardware = detect_hardware()
        assert hardware["gpu"]["type"] in ("nvidia", "none")


class TestBenchmarkMemory:
    def test_nonexistent_file_raises(self, tmp_path: Path):
        """benchmark_memory raises FileNotFoundError for nonexistent file."""
        model_file = tmp_path / "nonexistent.gguf"
        with pytest.raises(FileNotFoundError):
            benchmark_memory(model_path=model_file)

    def test_non_file_raises(self, tmp_path: Path):
        """benchmark_memory raises FileNotFoundError for directory."""
        with pytest.raises(FileNotFoundError):
            benchmark_memory(model_path=tmp_path)

    def test_returns_dict(self, tmp_path: Path, mock_model_file: Path):
        """benchmark_memory returns dict with required keys."""
        result = benchmark_memory(model_path=mock_model_file)
        assert isinstance(result, dict)
        assert "vram_peak_gb" in result
        assert "cpu_ram_peak_gb" in result
        assert "model_file_size_gb" in result
        assert result["model_file_size_gb"] >= 0  # File exists, size >= 0
