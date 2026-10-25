"""Tests for fairbench.reporter module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fairbench.reporter import Reporter


class TestReporter:
    def test_generate_report(self, tmp_results_dir: Path):
        """test_generate_report: generate_report creates a JSON file with correct structure."""
        reporter = Reporter(results_dir=tmp_results_dir)
        hardware = {
            "cpu": {"model": "Test", "cores": 4, "threads": 8},
            "ram_gb": 16.0,
            "gpu": {"type": "none"},
            "os": "Linux",
        }
        report_path = reporter.generate_report(
            model_name="test-model",
            quantization="q4_k_m",
            hardware=hardware,
            quality={"mmlu": 0.75},
            speed={"ttft_ms": 100.0, "tpot_ms": 5.0, "throughput": 50.0, "measured": True},
            memory={"vram_peak_gb": 4.0, "cpu_ram_peak_gb": 8.0, "model_file_size_gb": 3.5},
            publish=False,
        )

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Read and verify content
        content = json.loads(report_path.read_text())
        assert content["model"]["name"] == "test-model"
        assert content["model"]["quantization"] == "q4_k_m"
        assert content["quality"]["mmlu"] == 0.75
        assert content["published"] is False
        assert "hardware" in content
        assert "speed" in content
        assert "memory" in content

    def test_list_reports(self, tmp_results_dir: Path):
        """test_list_reports: list_reports returns list of report summaries."""
        reporter = Reporter(results_dir=tmp_results_dir)

        # Generate a report first
        hardware = {
            "cpu": {"model": "Test", "cores": 4, "threads": 8},
            "ram_gb": 16.0,
            "gpu": {"type": "none"},
            "os": "Linux",
        }
        reporter.generate_report(
            model_name="test-model",
            quantization="q4_k_m",
            hardware=hardware,
            quality={"mmlu": 0.75},
            speed={"ttft_ms": 100.0, "tpot_ms": 5.0, "throughput": 50.0, "measured": True},
            memory={"vram_peak_gb": 4.0, "cpu_ram_peak_gb": 8.0, "model_file_size_gb": 3.5},
            publish=False,
        )

        reports = reporter.list_reports()
        assert len(reports) >= 1
        assert "id" in reports[0]
        assert "timestamp" in reports[0]
        assert "model" in reports[0]
        assert "published" in reports[0]

    def test_get_latest(self, tmp_results_dir: Path):
        """test_get_latest: get_latest returns the most recent report."""
        reporter = Reporter(results_dir=tmp_results_dir)

        # Generate two reports
        hardware = {
            "cpu": {"model": "Test", "cores": 4, "threads": 8},
            "ram_gb": 16.0,
            "gpu": {"type": "none"},
            "os": "Linux",
        }

        reporter.generate_report(
            model_name="model-a",
            quantization="q4_k_m",
            hardware=hardware,
            quality={"mmlu": 0.7},
            speed={"ttft_ms": 100.0, "tpot_ms": 5.0, "throughput": 50.0, "measured": True},
            memory={"vram_peak_gb": 4.0, "cpu_ram_peak_gb": 8.0, "model_file_size_gb": 3.5},
            publish=False,
        )

        import time
        time.sleep(0.1)  # Ensure different timestamp

        reporter.generate_report(
            model_name="model-b",
            quantization="q4_k_m",
            hardware=hardware,
            quality={"mmlu": 0.8},
            speed={"ttft_ms": 80.0, "tpot_ms": 4.0, "throughput": 60.0, "measured": True},
            memory={"vram_peak_gb": 3.5, "cpu_ram_peak_gb": 7.5, "model_file_size_gb": 3.0},
            publish=False,
        )

        latest = reporter.get_latest()
        assert latest is not None
        assert latest["model"]["name"] == "model-b"

    def test_get_latest_no_reports(self, tmp_results_dir: Path):
        """test_get_latest_no_reports: get_latest returns None when no reports exist."""
        reporter = Reporter(results_dir=tmp_results_dir)
        latest = reporter.get_latest()
        assert latest is None

    @patch("fairbench.reporter.requests.post")
    def test_publish_calls_api(self, mock_post, tmp_results_dir: Path):
        """test_publish_calls_api: publish sends POST to API."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        reporter = Reporter(results_dir=tmp_results_dir)
        hardware = {
            "cpu": {"model": "Test", "cores": 4, "threads": 8},
            "ram_gb": 16.0,
            "gpu": {"type": "none"},
            "os": "Linux",
        }
        reporter.generate_report(
            model_name="test-model",
            quantization="q4_k_m",
            hardware=hardware,
            quality={"mmlu": 0.75},
            speed={"ttft_ms": 100.0, "tpot_ms": 5.0, "throughput": 50.0, "measured": True},
            memory={"vram_peak_gb": 4.0, "cpu_ram_peak_gb": 8.0, "model_file_size_gb": 3.5},
            publish=True,
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/report" in call_args.args[0]
