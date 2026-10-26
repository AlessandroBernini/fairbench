"""Tests for fairbench.reporter module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fairbench.config import BenchmarkConfig
from fairbench.reporter import Reporter


@pytest.fixture
def mock_config(tmp_results_dir: Path):
    """Create a mock config for Reporter tests."""
    return BenchmarkConfig(
        api_url="http://localhost:3000/api",
        default_n_ctx=4096,
        default_n_gpu_layers=-1,
        default_quant="q4_k_m",
        default_threads=0,
        publish=False,
        results_dir=str(tmp_results_dir),
    )


class TestReporter:
    def test_save_local_report(self, tmp_results_dir: Path, mock_config: BenchmarkConfig):
        """test_save_local_report: save_local_report creates a JSON file with correct structure."""
        reporter = Reporter(config=mock_config)
        result = {
            "model_name": "test-model",
            "model_path": "/path/to/model.gguf",
            "quantization": "q4_k_m",
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "n_threads": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "category": "quality",
            "quality_results": [
                {
                    "benchmark_name": "mmlu",
                    "score": 0.75,
                    "items_total": 100,
                    "items_correct": 75,
                    "metadata": {},
                }
            ],
            "speed_result": {
                "ttft_avg": 100.0,
                "tpot_avg": 5.0,
                "throughput_avg": 50.0,
                "runs": 3,
                "metadata": {},
            },
            "memory_result": {"vram_peak_mb": 4000.0, "ram_peak_mb": 8000.0, "metadata": {}},
            "swebench_results": [],
        }
        report_path = reporter.save_local_report("test-run", result)

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Read and verify content
        content = json.loads(report_path.read_text())
        assert content["model_name"] == "test-model"
        assert content["quantization"] == "q4_k_m"

    def test_format_console_output(self, mock_config: BenchmarkConfig):
        """format_console_output returns a formatted string."""
        reporter = Reporter(config=mock_config)
        result = {
            "model_name": "test-model",
            "model_path": "/path/to/model.gguf",
            "quantization": "q4_k_m",
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "n_threads": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "category": "quality",
            "quality_results": [
                {
                    "benchmark_name": "mmlu",
                    "score": 0.75,
                    "items_total": 100,
                    "items_correct": 75,
                    "metadata": {},
                }
            ],
            "speed_result": {
                "ttft_avg": 100.0,
                "tpot_avg": 5.0,
                "throughput_avg": 50.0,
                "runs": 3,
                "metadata": {},
            },
            "memory_result": {"vram_peak_mb": 4000.0, "ram_peak_mb": 8000.0, "metadata": {}},
            "swebench_results": [],
        }
        output = reporter.format_console_output("test-run", "test-model", "quality", result)
        assert "FairBench Result" in output
        assert "test-model" in output
        assert "mmlu" in output

    def test_publish_false_returns_none(self, mock_config: BenchmarkConfig):
        """publish_result returns None when publish is False."""
        reporter = Reporter(config=mock_config)
        result = {"model_name": "test-model"}
        assert reporter.publish_result("test-run", result) is None

    @patch("fairbench.reporter.requests.post")
    def test_publish_calls_api(self, mock_post, mock_config: BenchmarkConfig):
        """publish_result sends POST to API when publish is True."""
        mock_config.publish = True
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_response

        reporter = Reporter(config=mock_config)
        result = {
            "model_name": "test-model",
            "model_path": "/path/to/model.gguf",
            "quantization": "q4_k_m",
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "n_threads": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "category": "quality",
            "quality_scores": {"mmlu": 0.75},
            "speed": {
                "ttft_ms": 100.0,
                "tpot_tokens_per_sec": 5.0,
                "throughput_tokens_per_sec": 50.0,
            },
            "memory": {"vram_peak_mb": 4000.0, "ram_peak_mb": 8000.0},
            "swebench": {"resolved": 0, "total": 0},
        }
        response = reporter.publish_result("test-run", result)

        mock_post.assert_called_once()
        assert response == {"status": "ok"}

    def test_get_leaderboard(self, mock_config: BenchmarkConfig):
        """get_leaderboard returns results from API."""
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {"model_name": "model-1", "quality_scores": {"mmlu": 0.7}},
                {"model_name": "model-2", "quality_scores": {"mmlu": 0.8}},
            ]
            mock_get.return_value = mock_response

            results = reporter.get_leaderboard(limit=10)
            assert len(results) == 2
            assert results[0]["model_name"] == "model-1"
