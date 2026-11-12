"""Tests for fairbench.compare — benchmark comparison engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from fairbench.compare import ComparisonEngine


# --- Fixtures ---

def _write_report(tmpdir: Path, run_id: str, data: dict[str, Any]) -> Path:
    """Write a fake _report.json file to tmpdir."""
    filepath = tmpdir / f"{run_id}_report.json"
    filepath.write_text(json.dumps(data))
    return filepath


SAMPLE_QUALITY = [
    {"benchmark_name": "mmlu", "score": 68.2, "items_total": 100, "items_correct": 68},
    {"benchmark_name": "hellaswag", "score": 74.1, "items_total": 100, "items_correct": 74},
]

SAMPLE_SPEED = {"ttft_avg": 12.5, "tpot_avg": 0.045, "throughput_avg": 22.2}

SAMPLE_HARDWARE = {
    "cpu_model": "Apple M2",
    "cpu_cores": 8,
    "cpu_threads": 8,
    "ram_gb": 16.0,
    "gpu_type": "apple-silicon",
    "gpu_model": "Apple M2 GPU",
    "gpu_vram_gb": 16.0,
    "os": "darwin",
    "n_ctx": 4096,
    "n_gpu_layers": -1,
}


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a temporary results directory with sample reports."""
    d = tmp_path / "results"
    d.mkdir()

    # Report 1: qwen2.5-7b-q4_k_m
    _write_report(
        d,
        "run_001",
        {
            "run_id": "run_001",
            "model_name": "qwen2.5-7b-q4_k_m",
            "category": "quality",
            "hardware": SAMPLE_HARDWARE,
            "quality_results": SAMPLE_QUALITY,
            "speed_result": SAMPLE_SPEED,
        },
    )

    # Report 2: llama3.1-8b-q4_k_m
    _write_report(
        d,
        "run_002",
        {
            "run_id": "run_002",
            "model_name": "llama3.1-8b-q4_k_m",
            "category": "quality",
            "hardware": SAMPLE_HARDWARE,
            "quality_results": [
                {"benchmark_name": "mmlu", "score": 65.7, "items_total": 100, "items_correct": 66},
                {"benchmark_name": "hellaswag", "score": 72.8, "items_total": 100, "items_correct": 73},
            ],
            "speed_result": SAMPLE_SPEED,
        },
    )

    # Report 3: mistral-7b-q5_k_m
    _write_report(
        d,
        "run_003",
        {
            "run_id": "run_003",
            "model_name": "mistral-7b-q5_k_m",
            "category": "quality",
            "hardware": {**SAMPLE_HARDWARE, "ram_gb": 32.0},
            "quality_results": [
                {"benchmark_name": "mmlu", "score": 67.1, "items_total": 100, "items_correct": 67},
                {"benchmark_name": "hellaswag", "score": 71.5, "items_total": 100, "items_correct": 72},
            ],
            "speed_result": SAMPLE_SPEED,
        },
    )

    # Report 4: qwen2.5-7b-q6_k (different quantization, same model base)
    _write_report(
        d,
        "run_004",
        {
            "run_id": "run_004",
            "model_name": "qwen2.5-7b-q6_k",
            "category": "quality",
            "hardware": SAMPLE_HARDWARE,
            "quality_results": [
                {"benchmark_name": "mmlu", "score": 70.5, "items_total": 100, "items_correct": 71},
                {"benchmark_name": "hellaswag", "score": 75.3, "items_total": 100, "items_correct": 75},
            ],
            "speed_result": SAMPLE_SPEED,
        },
    )

    return d


# --- Tests: ComparisonEngine ---

class TestComparisonEngineInit:
    def test_init_with_no_results_dir(self):
        """Engine can be initialized even when results_dir doesn't exist."""
        engine = ComparisonEngine(results_dir="/tmp/nonexistent_fairbench_results_xyz")
        assert engine.results_dir is not None

    def test_compare_empty_dir(self, results_dir):
        """compare_runs returns empty result when no matching results exist."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs(models=["nonexistent*"])
        assert result.models == []
        assert result.benchmarks == {}


class TestCompareRuns:
    def test_compare_all_results(self, results_dir):
        """compare_runs aggregates all results by default."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()

        assert len(result.models) == 4
        assert "mmlu" in result.benchmarks
        assert "hellaswag" in result.benchmarks
        assert len(result.run_ids) == 4

    def test_compare_by_run_ids(self, results_dir):
        """compare_runs filters by specific run IDs."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs(run_ids=["run_001", "run_002"])

        assert len(result.models) == 2
        assert "qwen2.5-7b-q4_k_m" in result.models
        assert "llama3.1-8b-q4_k_m" in result.models

    def test_compare_by_model_pattern(self, results_dir):
        """compare_runs filters by model name pattern."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs(models=["qwen*"])

        assert len(result.models) == 2
        assert all("qwen" in m for m in result.models)

    def test_compare_by_category(self, results_dir):
        """compare_runs filters by benchmark category."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs(categories=["quality"])

        assert len(result.models) == 4  # All reports are quality

    def test_compare_combined_filters(self, results_dir):
        """compare_runs applies multiple filters together."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs(
            models=["qwen*"],
            categories=["quality"],
        )

        assert len(result.models) == 2
        assert "qwen2.5-7b-q4_k_m" in result.models
        assert "qwen2.5-7b-q6_k" in result.models


class TestAggregate:
    def test_benchmark_scores_aggregated(self, results_dir):
        """Aggregate collects scores per benchmark across models."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()

        # qwen2.5-7b-q4_k_m has mmlu=68.2, qwen2.5-7b-q6_k has mmlu=70.5
        assert 68.2 in result.benchmarks["mmlu"]
        assert 70.5 in result.benchmarks["mmlu"]

    def test_stats_computed(self, results_dir):
        """Stats dict contains mean, stdev, min, max, count."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()

        # Check stats exist for at least one model/benchmark combo
        for model in result.models:
            for bench in result.benchmarks:
                key = f"{model}::{bench}"
                if key in result.stats:
                    stat = result.stats[key]
                    assert "mean" in stat
                    assert "stdev" in stat
                    assert "min" in stat
                    assert "max" in stat
                    assert "count" in stat
                    assert stat["count"] >= 1
                    break
            else:
                continue
            break


class TestConsoleTable:
    def test_empty_result(self):
        """Empty comparison returns informative message."""
        engine = ComparisonEngine()
        result = engine.compare_runs()
        output = result.to_console_table()
        assert "No results" in output

    def test_table_has_header(self, results_dir):
        """Console table includes model column header."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()
        output = result.to_console_table()

        assert "Model" in output
        assert "─" in output  # separator line

    def test_table_shows_scores(self, results_dir):
        """Console table displays numerical scores."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()
        output = result.to_console_table()

        assert "qwen2.5-7b-q4_k_m" in output
        assert "llama3.1-8b-q4_k_m" in output
        assert "68.20" in output  # mmlu mean for qwen2.5-7b-q4_k_m


class TestCSVOutput:
    def test_csv_has_header(self, results_dir):
        """CSV output includes column headers."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()
        output = result.to_csv()

        assert "model" in output
        assert "mmlu" in output

    def test_csv_empty(self):
        """CSV output is empty string when no results."""
        engine = ComparisonEngine()
        result = engine.compare_runs()
        output = result.to_csv()
        assert output == ""


class TestJSONOutput:
    def test_json_contains_models(self, results_dir):
        """JSON output includes model list."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()
        output = result.to_json()
        data = json.loads(output)

        assert "models" in data
        assert len(data["models"]) == 4

    def test_json_contains_stats(self, results_dir):
        """JSON output includes stats dict."""
        engine = ComparisonEngine(results_dir=str(results_dir))
        result = engine.compare_runs()
        output = result.to_json()
        data = json.loads(output)

        assert "stats" in data
        assert isinstance(data["stats"], dict)


class TestCompareEdgeCases:
    def test_corrupted_json_file(self, tmp_path: Path):
        """Engine skips corrupted JSON files without crashing."""
        d = tmp_path / "results"
        d.mkdir()

        # Write valid report
        _write_report(
            d,
            "run_valid",
            {
                "run_id": "run_valid",
                "model_name": "test-model",
                "category": "quality",
                "quality_results": [{"benchmark_name": "mmlu", "score": 50.0}],
            },
        )

        # Write corrupted file
        (d / "run_corrupted_report.json").write_text("NOT VALID JSON {{{")

        engine = ComparisonEngine(results_dir=str(d))
        result = engine.compare_runs()

        assert len(result.models) == 1
        assert "test-model" in result.models

    def test_missing_hardware_data(self, tmp_path: Path):
        """Engine handles results with missing hardware data."""
        d = tmp_path / "results"
        d.mkdir()

        _write_report(
            d,
            "run_no_hw",
            {
                "run_id": "run_no_hw",
                "model_name": "test-model",
                "category": "quality",
                "quality_results": [{"benchmark_name": "mmlu", "score": 50.0}],
            },
        )

        engine = ComparisonEngine(results_dir=str(d))
        result = engine.compare_runs()

        assert len(result.models) == 1

    def test_compare_with_custom_filter(self, results_dir):
        """compare_runs respects custom filter_fn."""
        engine = ComparisonEngine(results_dir=str(results_dir))

        def high_mmlu(result: dict[str, Any]) -> bool:
            for qr in result.get("quality_results", []):
                if qr["benchmark_name"] == "mmlu" and qr["score"] > 67:
                    return True
            return False

        result = engine.compare_runs(filter_fn=high_mmlu)

        # Only models with mmlu > 67
        assert len(result.models) >= 2
        assert "llama3.1-8b-q4_k_m" not in result.models  # mmlu=65.7
