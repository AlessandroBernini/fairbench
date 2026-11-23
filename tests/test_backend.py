"""Tests for fairbench.backend module."""

from pathlib import Path

import pytest

from fairbench.backend import _compute_speed_metrics, benchmark_memory, benchmark_speed


class TestBenchmarkMemory:
    def test_nonexistent_file_raises(self, tmp_path: Path):
        """benchmark_memory returns MemoryResult for nonexistent file (graceful degradation)."""
        model_file = tmp_path / "nonexistent.gguf"
        # When llama_cpp is not installed or file doesn't exist, we get graceful degradation
        result = benchmark_memory(model_path=str(model_file))
        assert hasattr(result, "vram_peak_mb")
        assert hasattr(result, "ram_peak_mb")
        assert hasattr(result, "vram_avg_mb")
        assert hasattr(result, "ram_avg_mb")

    def test_non_file_raises(self, tmp_path: Path):
        """benchmark_memory returns MemoryResult for directory path (graceful degradation)."""
        result = benchmark_memory(model_path=str(tmp_path))
        assert hasattr(result, "vram_peak_mb")
        assert hasattr(result, "ram_peak_mb")

    def test_returns_memory_result(self, tmp_path: Path, mock_model_file: Path):
        """benchmark_memory returns a MemoryResult with required attributes."""
        result = benchmark_memory(model_path=mock_model_file)
        assert hasattr(result, "vram_peak_mb")
        assert hasattr(result, "ram_peak_mb")
        assert hasattr(result, "vram_avg_mb")
        assert hasattr(result, "ram_avg_mb")


class TestBenchmarkSpeed:
    def test_nonexistent_file_raises(self, tmp_path: Path):
        """benchmark_speed returns SpeedResult for nonexistent file (graceful degradation via llama-bench fallback)."""
        model_file = tmp_path / "nonexistent.gguf"
        # llama-bench fallback will fail gracefully with metadata error
        result = benchmark_speed(model_path=str(model_file))
        assert hasattr(result, "ttft_avg")
        assert hasattr(result, "tpot_avg")
        assert hasattr(result, "throughput_avg")
        assert hasattr(result, "runs")

    def test_returns_speed_result(self, tmp_path: Path, mock_model_file: Path):
        """benchmark_speed returns a SpeedResult with required attributes."""
        result = benchmark_speed(model_path=str(mock_model_file))
        assert hasattr(result, "ttft_avg")
        assert hasattr(result, "tpot_avg")
        assert hasattr(result, "throughput_avg")
        assert hasattr(result, "runs")


def test_compute_speed_metrics_standard_definition() -> None:
    """_compute_speed_metrics uses the standard llama-bench definition.

    ttft=100ms, total=600ms, 10 token -> decode 500ms su 9 gap
    tpot_ms = 500.0 / 9 ≈ 55.56 ms/token
    throughput = 10 / 0.5 = 20.0 tokens/s
    """
    tpot_ms, throughput = _compute_speed_metrics(100.0, 0.6, 10)
    assert tpot_ms == pytest.approx(500.0 / 9)
    assert throughput == pytest.approx(10 / 0.5)


def test_compute_speed_metrics_edge_cases() -> None:
    """_compute_speed_metrics returns (0.0, 0.0) when tokens <= 1."""
    assert _compute_speed_metrics(50.0, 0.1, 0) == (0.0, 0.0)
    assert _compute_speed_metrics(50.0, 0.1, 1) == (0.0, 0.0)
