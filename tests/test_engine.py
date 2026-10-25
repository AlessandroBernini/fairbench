"""Tests for fairbench.engine module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fairbench.engine import run_benchmark
from fairbench.quality import DEFAULT_TASKS


class TestRunBenchmark:
    def test_default_tasks_backward_compat(self, mock_model_file: Path, mock_reporter):
        """run_benchmark uses DEFAULT_TASKS when tasks not specified."""
        with patch("fairbench.engine.detect_hardware") as mock_hw, \
             patch("fairbench.engine.run_quality_benchmark") as mock_quality, \
             patch("fairbench.engine.benchmark_speed") as mock_speed, \
             patch("fairbench.engine.benchmark_memory") as mock_mem, \
             patch("fairbench.engine.Reporter") as mock_reporter_class:

            mock_hw.return_value = {
                "cpu": {"model": "Test", "cores": 4, "threads": 8},
                "ram_gb": 16.0,
                "gpu": {"type": "none"},
                "os": "Linux",
            }
            mock_quality.return_value = {"mmlu": 0.5}
            mock_speed.return_value = {
                "ttft_ms": 100.0, "tpot_ms": 5.0,
                "throughput": 50.0, "measured": True
            }
            mock_mem.return_value = {
                "vram_peak_gb": 4.0,
                "cpu_ram_peak_gb": 8.0,
                "model_file_size_gb": 3.5
            }

            mock_reporter_instance = MagicMock()
            mock_reporter_instance.generate_report.return_value = Path("/tmp/report.json")
            mock_reporter_class.return_value = mock_reporter_instance

            result = run_benchmark(
                model_path=mock_model_file,
                model_name="test-model",
                tasks=None,  # Should use DEFAULT_TASKS
            )

            assert result == Path("/tmp/report.json")
            # Verify quality was called with default tasks
            mock_quality.assert_called_once()
            call_kwargs = mock_quality.call_args
            # engine.py converts None to DEFAULT_TASKS, so expect DEFAULT_TASKS
            assert call_kwargs[1].get("task_names_in") == DEFAULT_TASKS

    def test_custom_tasks(self, mock_model_file: Path, mock_reporter):
        """run_benchmark uses custom tasks when specified."""
        with patch("fairbench.engine.detect_hardware") as mock_hw, \
             patch("fairbench.engine.run_quality_benchmark") as mock_quality, \
             patch("fairbench.engine.benchmark_speed") as mock_speed, \
             patch("fairbench.engine.benchmark_memory") as mock_mem, \
             patch("fairbench.engine.Reporter") as mock_reporter_class:

            mock_hw.return_value = {
                "cpu": {"model": "Test", "cores": 4, "threads": 8},
                "ram_gb": 16.0,
                "gpu": {"type": "none"},
                "os": "Linux",
            }
            mock_quality.return_value = {"humaneval": 0.3, "ifeval": 0.7}
            mock_speed.return_value = {
                "ttft_ms": 100.0, "tpot_ms": 5.0,
                "throughput": 50.0, "measured": True
            }
            mock_mem.return_value = {
                "vram_peak_gb": 4.0,
                "cpu_ram_peak_gb": 8.0,
                "model_file_size_gb": 3.5
            }

            mock_reporter_instance = MagicMock()
            mock_reporter_instance.generate_report.return_value = Path("/tmp/report.json")
            mock_reporter_class.return_value = mock_reporter_instance

            result = run_benchmark(
                model_path=mock_model_file,
                model_name="test-model",
                tasks=["humaneval", "ifeval"],
            )

            assert result == Path("/tmp/report.json")
            mock_quality.assert_called_once()

    def test_publish_false(self, mock_model_file: Path, mock_reporter):
        """run_benchmark respects publish=False."""
        with patch("fairbench.engine.detect_hardware") as mock_hw, \
             patch("fairbench.engine.run_quality_benchmark") as mock_quality, \
             patch("fairbench.engine.benchmark_speed") as mock_speed, \
             patch("fairbench.engine.benchmark_memory") as mock_mem, \
             patch("fairbench.engine.Reporter") as mock_reporter_class:

            mock_hw.return_value = {
                "cpu": {"model": "Test", "cores": 4, "threads": 8},
                "ram_gb": 16.0,
                "gpu": {"type": "none"},
                "os": "Linux",
            }
            mock_quality.return_value = {"mmlu": 0.5}
            mock_speed.return_value = {
                "ttft_ms": 100.0, "tpot_ms": 5.0,
                "throughput": 50.0, "measured": True
            }
            mock_mem.return_value = {
                "vram_peak_gb": 4.0,
                "cpu_ram_peak_gb": 8.0,
                "model_file_size_gb": 3.5
            }

            mock_reporter_instance = MagicMock()
            mock_reporter_instance.generate_report.return_value = Path("/tmp/report.json")
            mock_reporter_class.return_value = mock_reporter_instance

            result = run_benchmark(
                model_path=mock_model_file,
                model_name="test-model",
                publish=False,
            )

            assert result == Path("/tmp/report.json")
            # Verify publish=False was passed
            mock_reporter_instance.generate_report.assert_called_once()
            assert mock_reporter_instance.generate_report.call_args[1].get("publish") is False
