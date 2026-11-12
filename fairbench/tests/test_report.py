"""Tests for fairbench.report — HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fairbench.report import ReportEngine


# --- Fixtures ---

def _create_test_run(tmp_path: Path) -> Path:
    """Create a minimal test run file."""
    results_dir = tmp_path / "results" / "run"
    results_dir.mkdir(parents=True)

    run_data: dict[str, Any] = {
        "model_name": "qwen2.5-7b-q4_k_m",
        "quantization": "q4_k_m",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_threads": 12,
        "timestamp": "2026-11-12T10:00:00Z",
        "quality_benchmarks": {
            "mmlu": {"score": 0.682, "items_correct": 3890, "items_total": 5700},
            "hellaswag": {"score": 0.741, "items_correct": 7410, "items_total": 10000},
        },
        "speed": {
            "throughput_tokens_per_sec": 45.2,
            "ttft_ms": 120.5,
            "tpot_ms": 22.1,
        },
        "memory": {
            "vram_peak_mb": 4500,
            "ram_mb": 2100,
        },
        "swebench": {
            "resolved": 0,
            "failed": 0,
            "errors": 0,
            "total": 0,
            "score": 0,
        },
    }

    run_file = results_dir / "run_12345.json"
    run_file.write_text(json.dumps(run_data), encoding="utf-8")

    return results_dir


class TestHTMLReport:
    def test_html_contains_model_name(self, tmp_path):
        """HTML report includes model name."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        html = engine.generate_html("run_12345")
        assert "qwen2.5-7b-q4_k_m" in html

    def test_html_contains_chart(self, tmp_path):
        """HTML report includes Chart.js canvas."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        html = engine.generate_html("run_12345")
        assert "qualityChart" in html
        assert "chart.js" in html

    def test_html_contains_speed_stats(self, tmp_path):
        """HTML report includes speed statistics."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        html = engine.generate_html("run_12345")
        assert "45.2" in html  # throughput
        assert "120.5" in html  # ttft

    def test_html_contains_swebench(self, tmp_path):
        """HTML report includes SWE-bench section."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        html = engine.generate_html("run_12345")
        assert "SWE-bench" in html
        assert "swebench-table" in html

    def test_save_report_creates_file(self, tmp_path):
        """save_report creates HTML file in reports directory."""
        results_dir = _create_test_run(tmp_path)
        reports_dir = tmp_path / "reports"
        engine = ReportEngine(results_dir=str(results_dir))
        engine.reports_dir = reports_dir  # Override for test

        report_path = engine.save_report("run_12345")

        assert report_path.exists()
        assert report_path.name == "run_12345_report.html"
        assert report_path.parent == reports_dir

    def test_report_file_valid_html(self, tmp_path):
        """Generated HTML is valid (has DOCTYPE, html, body tags)."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        report_path = engine.save_report("run_12345")
        html = report_path.read_text()

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<body>" in html
        assert "</html>" in html


class TestReportErrors:
    def test_missing_run_file(self, tmp_path):
        """generate_html raises FileNotFoundError for missing run."""
        results_dir = _create_test_run(tmp_path)
        engine = ReportEngine(results_dir=str(results_dir))

        with pytest.raises(FileNotFoundError, match="Run file not found"):
            engine.generate_html("nonexistent_run")
