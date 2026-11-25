"""Tests for the FairBench report upload API contract.

Covers the integration between the local reporter (fairbench.reporter) and the
Vercel serverless report APIs (POST /api/report, GET /api/leaderboard).

Contract under test:
- POST /api/report receives a flat payload with the columns expected by the
  SQL INSERT in api/report.ts.
- GET /api/leaderboard returns an envelope of the shape { "reports": [...] }.
"""

from unittest.mock import MagicMock, patch

import pytest

from fairbench.config import BenchmarkConfig
from fairbench.reporter import Reporter


@pytest.fixture
def mock_config(tmp_results_dir):
    return BenchmarkConfig(
        api_url="http://localhost:3000/api",
        default_n_ctx=4096,
        default_n_gpu_layers=-1,
        default_quant="q4_k_m",
        default_threads=0,
        publish=True,
        results_dir=str(tmp_results_dir),
    )


def _engine_style_result() -> dict:
    """Result dict as produced by fairbench.engine.BenchmarkEngine.run_benchmark."""
    return {
        "run_id": "checkpoint_1764000000",
        "model_name": "qwen2.5-7b-instruct",
        "model_path": "/home/jarvispi/models/qwen2.5-7b-instruct.gguf",
        "quantization": "q4_k_m",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_threads": 8,
        "timestamp": "2026-11-25T11:40:00.123456+00:00",
        "category": "quality",
        "quality_results": [
            {
                "benchmark_name": "mmlu",
                "score": 0.623,
                "items_total": 5700,
                "items_correct": 3890,
                "metadata": {},
            },
            {
                "benchmark_name": "hellaswag",
                "score": 0.834,
                "items_total": 10000,
                "items_correct": 8340,
                "metadata": {},
            },
            {
                "benchmark_name": "gsm8k",
                "score": 0.712,
                "items_total": 880,
                "items_correct": 622,
                "metadata": {},
            },
        ],
        "speed_result": {
            "ttft_avg": 145.2,
            "tpot_avg": 25.3,
            "throughput_avg": 35.2,
            "runs": 3,
            "metadata": {},
        },
        "memory_result": {
            "vram_peak_mb": 4500.0,
            "ram_peak_mb": 8192.0,
            "metadata": {},
        },
        "swebench_results": [],
        "hardware_profile_hash": "abc123",
        "hardware_snapshot": {
            "cpu_model": "AMD Ryzen 9",
            "cpu_cores": 12,
            "cpu_threads": 24,
            "total_ram_gb": 64.0,
            "gpu_available": True,
            "gpu_type": "nvidia",
            "gpu_vram_gb": 12.0,
            "os_platform": "Linux",
            "os_version": "6.1",
            "kernel_version": "6.1.0",
        },
    }


def _mock_ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = payload
    return resp


class TestReportUploadApi:
    def test_publish_posts_to_report_endpoint(self, mock_config: BenchmarkConfig):
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.post") as mock_post:
            mock_post.return_value = _mock_ok_response(
                {"success": True, "report": {"id": "checkpoint_1764000000", "published": True}}
            )

            response = reporter.publish_result("checkpoint_1764000000", _engine_style_result())

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "http://localhost:3000/api/report"
            assert kwargs["json"]["run_id"] == "checkpoint_1764000000"
            assert response["success"] is True

    def test_publish_payload_contains_api_required_fields(self, mock_config: BenchmarkConfig):
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.post") as mock_post:
            mock_post.return_value = _mock_ok_response({"success": True, "report": {"id": "x"}})

            reporter.publish_result("checkpoint_1764000000", _engine_style_result())

            payload = mock_post.call_args.kwargs["json"]
            # Required by the INSERT in api/report.ts
            assert payload["run_id"] == "checkpoint_1764000000"
            assert payload["model_name"]
            assert payload["quantization"]
            assert payload["timestamp"]

    def test_publish_maps_memory_mb_to_gb_for_api(self, mock_config: BenchmarkConfig):
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.post") as mock_post:
            mock_post.return_value = _mock_ok_response({"success": True, "report": {"id": "x"}})

            reporter.publish_result("checkpoint_1764000000", _engine_style_result())

            payload = mock_post.call_args.kwargs["json"]
            assert payload["memory"]["vram_peak_gb"] > 0
            assert payload["memory"]["cpu_ram_peak_gb"] > 0
            # 4500 MB VRAM -> ~4.39 GB ; 8192 MB RAM -> ~8.0 GB
            assert abs(payload["memory"]["vram_peak_gb"] - 4500.0 / 1024) < 1e-6
            assert abs(payload["memory"]["cpu_ram_peak_gb"] - 8192.0 / 1024) < 1e-6

    def test_publish_includes_hardware_in_payload(self, mock_config: BenchmarkConfig):
        """The INSERT in api/report.ts requires hardware columns to be populated."""
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.post") as mock_post:
            mock_post.return_value = _mock_ok_response({"success": True, "report": {"id": "x"}})

            reporter.publish_result("checkpoint_1764000000", _engine_style_result())

            payload = mock_post.call_args.kwargs["json"]
            hardware = payload["hardware"]
            assert hardware.get("cpu_model") not in (None, "")
            assert hardware.get("gpu_type") not in (None, "")
            assert hardware.get("gpu_vram_gb") not in (None, 0)
            assert hardware.get("ram_gb") not in (None, 0)


class TestLeaderboardApi:
    def test_get_leaderboard_hits_leaderboard_endpoint(self, mock_config: BenchmarkConfig):
        reporter = Reporter(config=mock_config)
        with patch("fairbench.reporter.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            mock_get.return_value.json.return_value = {"reports": []}

            results = reporter.get_leaderboard(limit=20)

            assert mock_get.call_args.args[0] == "http://localhost:3000/api/leaderboard"
            assert results == []

    def test_get_leaderboard_unwraps_reports_envelope(self, mock_config: BenchmarkConfig):
        reporter = Reporter(config=mock_config)
        reports = [{"model_name": "qwen2.5-7b-instruct", "quality_scores": {"mmlu": 0.62}}]
        with patch("fairbench.reporter.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            mock_get.return_value.json.return_value = {"reports": reports}

            results = reporter.get_leaderboard(limit=20)

            assert results == reports
