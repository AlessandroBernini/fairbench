"""Tests for fairbench.backend module."""

from pathlib import Path

from fairbench.backend import benchmark_memory, benchmark_speed


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
