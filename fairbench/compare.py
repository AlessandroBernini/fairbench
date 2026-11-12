"""Benchmark results comparison engine."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fairbench.config import load_config


@dataclass
class ComparisonResult:
    """Aggregated comparison across multiple runs."""

    models: list[str] = field(default_factory=list)
    benchmarks: dict[str, list[float]] = field(default_factory=dict)
    hardware_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)

    def to_console_table(
        self,
        benchmarks: list[str] | None = None,
        sort_by: str | None = None,
    ) -> str:
        """Format comparison result as a console table."""
        if not self.models:
            return "No results to compare."

        # Determine which benchmarks to show
        if benchmarks is None:
            # Show quality benchmarks by default
            quality_benches = [
                b for b in self.benchmarks if b not in ("ttft_ms", "tpot_ms", "throughput")
            ]
            benchmarks = quality_benches if quality_benches else list(self.benchmarks.keys())

        # Sort models
        models = self.models
        if sort_by:
            models = sorted(
                models,
                key=lambda m: self.stats.get(f"{m}::{sort_by}", {}).get("mean", 0),
                reverse=True,
            )

        # Calculate column widths
        col_widths: dict[str, int] = {"Model": 30}
        for bench in benchmarks:
            scores = self.benchmarks.get(bench, [0.0])
            formatted = [f"{v:.2f}" for v in scores]
            col_widths[bench] = max(10, max(len(s) for s in formatted)) + 2

        # Build header
        header = f"{'Model':<{col_widths['Model']}}"
        for bench in benchmarks:
            header += f"{bench:^{col_widths[bench]}}"
        header += "\n" + "─" * len(header)

        # Build rows
        rows = [header]
        for model in models:
            row = f"{model:<{col_widths['Model']}}"
            for bench in benchmarks:
                scores = self.benchmarks.get(bench, [])
                if scores:
                    mean_val = statistics.mean(scores)
                    row += f"{mean_val:^{col_widths[bench]}.2f}"
                else:
                    row += f"{'N/A':^{col_widths[bench]}}"
            rows.append(row)

        # Add stats footer
        rows.append("")
        rows.append("Statistics (mean ± stdev):")
        for model in models:
            stats_parts = []
            for bench in benchmarks:
                key = f"{model}::{bench}"
                stat = self.stats.get(key, {})
                if stat:
                    mean = stat.get("mean", 0)
                    stdev = stat.get("stdev", 0)
                    stats_parts.append(f"{mean:.2f}±{stdev:.2f}")
                else:
                    stats_parts.append("N/A")
            rows.append(f"  {model}: {' '.join(stats_parts)}")

        # Add hardware info
        if self.hardware_profiles:
            rows.append("")
            rows.append("Hardware profiles:")
            for model, hw in self.hardware_profiles.items():
                parts = []
                if hw.get("cpu_model"):
                    parts.append(f"CPU={hw['cpu_model']}")
                if hw.get("cpu_cores"):
                    parts.append(f"{hw['cpu_cores']}c")
                if hw.get("ram_gb"):
                    parts.append(f"RAM={hw['ram_gb']}GB")
                if hw.get("gpu_type") and hw.get("gpu_model"):
                    parts.append(f"GPU={hw['gpu_model']}")
                elif hw.get("gpu_type"):
                    parts.append(f"GPU={hw['gpu_type']}")
                if hw.get("n_ctx"):
                    parts.append(f"ctx={hw['n_ctx']}")
                if hw.get("n_gpu_layers") is not None:
                    parts.append(f"layers={hw['n_gpu_layers']}")
                rows.append(f"  {model}: {' '.join(parts)}")

        return "\n".join(rows)

    def to_csv(self) -> str:
        """Format comparison result as CSV."""
        if not self.models:
            return ""

        benchmarks = list(self.benchmarks.keys())
        output = []
        header = ["model"] + benchmarks
        output.append(",".join(header))

        for model in self.models:
            row = [model]
            for bench in benchmarks:
                scores = self.benchmarks.get(bench, [])
                row.append(",".join(f"{s}" for s in scores))
            output.append(",".join(row))

        return "\n".join(output)

    def to_json(self) -> str:
        """Format comparison result as JSON."""
        return json.dumps(
            {
                "models": self.models,
                "benchmarks": {
                    k: v for k, v in self.benchmarks.items()
                },
                "stats": self.stats,
                "hardware_profiles": self.hardware_profiles,
                "run_ids": self.run_ids,
            },
            indent=2,
            default=str,
        )


class ComparisonEngine:
    """Compare benchmark results across models, quantizations, or hardware."""

    def __init__(self, results_dir: str | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path(
            load_config().results_dir
        )

    def compare_runs(
        self,
        run_ids: list[str] | None = None,
        models: list[str] | None = None,
        categories: list[str] | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> ComparisonResult:
        """Compare benchmark results.

        Args:
            run_ids: Specific run IDs to compare.
            models: Model name patterns to filter (e.g. ["qwen*", "llama*"]).
            categories: Benchmark categories to include (e.g. ["quality", "speed"]).
            filter_fn: Custom filter function on result dicts.

        Returns:
            ComparisonResult with aggregated stats.
        """
        results = self._load_results(run_ids, models, categories, filter_fn)
        if not results:
            return ComparisonResult()

        return self._aggregate(results)

    def _load_results(
        self,
        run_ids: list[str] | None,
        models: list[str] | None,
        categories: list[str] | None,
        filter_fn: Callable[[dict[str, Any]], bool] | None,
    ) -> list[dict[str, Any]]:
        """Load and filter result files from disk."""
        if not self.results_dir.exists():
            return []

        results: list[dict[str, Any]] = []

        for filepath in sorted(self.results_dir.glob("*_report.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            run_id = data.get("run_id", filepath.stem.replace("_report", ""))

            # Filter by run IDs
            if run_ids and run_id not in run_ids:
                continue

            # Filter by model name patterns
            model_name = data.get("model_name", "")
            if models and not any(
                re.match(pattern.replace("*", ".*"), model_name, re.IGNORECASE)
                for pattern in models
            ):
                continue

            # Filter by category
            if categories:
                result_category = data.get("category", "")
                if result_category not in categories:
                    continue

            # Custom filter
            if filter_fn and not filter_fn(data):
                continue

            results.append(data)

        return results

    def _aggregate(self, results: list[dict[str, Any]]) -> ComparisonResult:
        """Aggregate results into a ComparisonResult."""
        comparison = ComparisonResult()

        # Collect unique benchmarks across all results
        benchmark_names: set[str] = set()
        for result in results:
            for qr in result.get("quality_results", []):
                benchmark_names.add(qr["benchmark_name"])
            speed = result.get("speed_result", {})
            if speed:
                benchmark_names.add("ttft_ms")
                benchmark_names.add("tpot_ms")
                benchmark_names.add("throughput")

        # Initialize benchmark data
        for bench in benchmark_names:
            comparison.benchmarks[bench] = []

        # Collect data per model
        model_data: dict[str, dict[str, list[float]]] = {}
        for result in results:
            model_name = result.get("model_name", "unknown")
            if model_name not in model_data:
                model_data[model_name] = {bench: [] for bench in benchmark_names}
                comparison.hardware_profiles[model_name] = result.get(
                    "hardware", {}
                )

            # Quality scores
            for qr in result.get("quality_results", []):
                name = qr["benchmark_name"]
                score = qr.get("score", 0.0)
                model_data[model_name][name].append(score)

            # Speed scores
            speed = result.get("speed_result", {})
            if speed:
                model_data[model_name]["ttft_ms"].append(speed.get("ttft_avg", 0))
                model_data[model_name]["tpot_ms"].append(speed.get("tpot_avg", 0))
                model_data[model_name]["throughput"].append(
                    speed.get("throughput_avg", 0)
                )

            comparison.run_ids.append(result.get("run_id", ""))

        # Build final benchmarks dict and stats
        for model_name, bench_data in model_data.items():
            comparison.models.append(model_name)
            for bench in benchmark_names:
                scores = bench_data[bench]
                comparison.benchmarks[bench].extend(scores)

                # Compute stats per model per benchmark
                if scores:
                    mean_val = statistics.mean(scores)
                    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
                    key = f"{model_name}::{bench}"
                    comparison.stats[key] = {
                        "mean": round(mean_val, 4),
                        "stdev": round(std_val, 4),
                        "min": round(min(scores), 4),
                        "max": round(max(scores), 4),
                        "count": len(scores),
                    }

        return comparison
