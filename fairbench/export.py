"""Export benchmark results to CSV and other formats."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


class ExportEngine:
    """Exports benchmark results to various formats."""

    def __init__(self, results_dir: str | Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path.home() / ".fairbench" / "results" / "run"

    def _load_results(self, run_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Load result files from disk."""
        if not self.results_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for filepath in sorted(self.results_dir.glob("*.json")):
            if run_ids and filepath.stem not in run_ids:
                continue
            try:
                with open(filepath, "r") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def export_csv(
        self,
        run_ids: list[str] | None = None,
        categories: list[str] | None = None,
        models: list[str] | None = None,
    ) -> str:
        """Export results to CSV format.

        Each benchmark becomes a row with config and scores.
        """
        results = self._load_results(run_ids=run_ids)

        # Filter by model patterns
        if models:
            results = [
                r for r in results
                if any(p.replace("*", "").lower() in r.get("model_name", "").lower() for p in models)
            ]

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "model_name", "quantization", "n_ctx", "n_gpu_layers", "n_threads",
            "timestamp", "benchmark", "score", "items_correct", "items_total",
            "ttft_ms", "tpot_ms", "vram_mb", "ram_mb", "resolved", "total_swebench"
        ])

        # Rows
        for data in results:
            model_name = data.get("model_name", "unknown")
            quantization = data.get("quantization", "")
            n_ctx = data.get("n_ctx", "")
            n_gpu_layers = data.get("n_gpu_layers", "")
            n_threads = data.get("n_threads", "")
            timestamp = data.get("timestamp", "")

            quality_benchmarks = data.get("quality_benchmarks", {})
            speed = data.get("speed", {})
            memory = data.get("memory", {})
            swebench = data.get("swebench", {})

            # Add one row per quality benchmark
            for bench_name, bench_data in quality_benchmarks.items():
                score = bench_data.get("score", 0.0) if isinstance(bench_data, dict) else bench_data
                items_correct = bench_data.get("items_correct", 0) if isinstance(bench_data, dict) else 0
                items_total = bench_data.get("items_total", 0) if isinstance(bench_data, dict) else 0

                row = [
                    model_name,
                    quantization,
                    n_ctx,
                    n_gpu_layers,
                    n_threads,
                    timestamp,
                    bench_name,
                    f"{score:.4f}",
                    items_correct,
                    items_total,
                    speed.get("ttft_ms", 0),
                    speed.get("tpot_ms", 0),
                    memory.get("vram_peak_mb", 0),
                    memory.get("ram_mb", 0),
                    swebench.get("resolved", 0),
                    swebench.get("total", 0),
                ]
                writer.writerow(row)

        return output.getvalue()

    def export_json(self, run_ids: list[str] | None = None) -> str:
        """Export results to JSON format."""
        results = self._load_results(run_ids=run_ids)
        return json.dumps(results, indent=2, default=str)
