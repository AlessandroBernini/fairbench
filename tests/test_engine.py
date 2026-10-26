"""Tests for fairbench.engine module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fairbench.engine import BenchmarkEngine
from fairbench.quality import AVAILABLE_BENCHMARKS
from fairbench.types import BenchmarkCategory


class TestBenchmarkEngine:
    def test_init_uses_default_config(self):
        """BenchmarkEngine initializes with default config when none provided."""
        with patch("fairbench.engine.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.results_dir = "/tmp/results"
            mock_load.return_value = mock_config

            engine = BenchmarkEngine()

            assert engine.config is not None
            mock_load.assert_called_once()

    def test_run_benchmark_calls_quality_benchmark(self, mock_model_file: Path):
        """run_benchmark calls run_quality_benchmark with correct parameters."""
        with (
            patch("fairbench.engine.load_config") as mock_load,
            patch("fairbench.engine.run_quality_benchmark") as mock_quality,
            patch("fairbench.engine.benchmark_speed") as mock_speed,
            patch("fairbench.engine.benchmark_memory") as mock_mem,
            patch("fairbench.engine.Reporter") as mock_reporter_class,
        ):
            mock_config = MagicMock()
            mock_config.results_dir = "/tmp/results"
            mock_config.publish = False
            mock_load.return_value = mock_config

            mock_speed.return_value = MagicMock(
                ttft_avg=100.0, tpot_avg=5.0, throughput_avg=50.0, runs=3, metadata={}
            )
            mock_mem.return_value = MagicMock(vram_peak_mb=4000.0, ram_peak_mb=8000.0, metadata={})

            mock_quality.return_value = [
                MagicMock(
                    benchmark_name="mmlu",
                    score=0.5,
                    items_total=100,
                    items_correct=50,
                    metadata={},
                )
            ]

            mock_reporter_instance = MagicMock()
            mock_reporter_class.return_value = mock_reporter_instance

            engine = BenchmarkEngine()
            result = engine.run_benchmark(
                model_path=str(mock_model_file),
                category=BenchmarkCategory.QUALITY,
            )

            assert result is not None
            mock_quality.assert_called_once()

    def test_list_available_benchmarks(self):
        """list_available_benchmarks returns AVAILABLE_BENCHMARKS."""
        with patch("fairbench.engine.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.results_dir = "/tmp/results"
            mock_load.return_value = mock_config

            engine = BenchmarkEngine()
            benchmarks = engine.list_available_benchmarks()

            assert benchmarks == AVAILABLE_BENCHMARKS

    def test_get_lite_benchmarks(self):
        """get_lite_benchmarks returns list of lite benchmarks."""
        with patch("fairbench.engine.load_config") as mock_load:
            mock_config = MagicMock()
            mock_config.results_dir = "/tmp/results"
            mock_load.return_value = mock_config

            engine = BenchmarkEngine()
            lite = engine.get_lite_benchmarks()

            assert isinstance(lite, list)
            assert len(lite) > 0
