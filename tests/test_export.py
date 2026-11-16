"""Tests for fairbench.export — CSV and JSON export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest

from fairbench.export import ExportEngine

# --- Fixtures ---

def _create_test_results(tmp_path: Path) -> Path:
    """Create minimal test results directory with sample data."""
    results_dir = tmp_path / "results" / "run"
    results_dir.mkdir(parents=True)

    # Sample run data
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


class TestCSVExport:
    def test_csv_header(self, tmp_path):
        """CSV output includes expected column headers."""
        results_dir = _create_test_results(tmp_path)
        engine = ExportEngine(results_dir=str(results_dir))
        output = engine.export_csv()

        reader = csv.reader(io.StringIO(output))
        header = next(reader)

        assert "model_name" in header
        assert "benchmark" in header
        assert "score" in header

    def test_csv_data_rows(self, tmp_path):
        """CSV output includes data rows for each benchmark."""
        results_dir = _create_test_results(tmp_path)
        engine = ExportEngine(results_dir=str(results_dir))
        output = engine.export_csv()

        reader = csv.reader(io.StringIO(output))
        next(reader)  # skip header
        rows = list(reader)

        assert len(rows) == 2  # mmlu + hellaswag
        assert rows[0][0] == "qwen2.5-7b-q4_k_m"
        assert rows[0][6] == "mmlu"
        assert float(rows[0][7]) == pytest.approx(0.682, abs=0.001)

    def test_csv_filter_by_run(self, tmp_path):
        """CSV export respects run_id filter."""
        results_dir = _create_test_results(tmp_path)
        engine = ExportEngine(results_dir=str(results_dir))
        output = engine.export_csv(run_ids=["run_12345"])

        assert "qwen2.5-7b-q4_k_m" in output

    def test_csv_empty_results(self):
        """CSV export returns header only when no results."""
        engine = ExportEngine(results_dir="/nonexistent")
        output = engine.export_csv()

        reader = csv.reader(io.StringIO(output))
        header = next(reader)
        assert len(header) > 0


class TestJSONExport:
    def test_json_structure(self, tmp_path):
        """JSON export preserves run data structure."""
        results_dir = _create_test_results(tmp_path)
        engine = ExportEngine(results_dir=str(results_dir))
        output = engine.export_json()

        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "model_name" in data[0]

    def test_json_empty(self):
        """JSON export returns empty list when no results."""
        engine = ExportEngine(results_dir="/nonexistent")
        output = engine.export_json()

        data = json.loads(output)
        assert data == []
