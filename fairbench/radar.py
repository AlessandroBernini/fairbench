"""Generate radar/spider charts for benchmark visualization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fairbench.compare import ComparisonEngine


class RadarEngine:
    """Generates radar/spider charts for benchmark runs."""

    def __init__(self, results_dir: str | Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path.home() / ".fairbench" / "results" / "run"

    def generate_radar(self, run_id: str) -> dict[str, Any]:
        """Generate radar chart data for a specific run."""
        run_file = self.results_dir / f"{run_id}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run file not found: {run_file}")

        with open(run_file, "r") as f:
            data = json.load(f)

        quality_benchmarks = data.get("quality_benchmarks", {})
        if not quality_benchmarks:
            return {"labels": [], "datasets": []}

        labels = list(quality_benchmarks.keys())
        scores = [v["score"] if isinstance(v, dict) else v for v in quality_benchmarks.values()]

        # Calculate average scores across all models for comparison
        engine = ComparisonEngine(results_dir=str(self.results_dir))
        comparison = engine.compare_runs()

        avg_scores = []
        for bench in labels:
            bench_scores = comparison.benchmarks.get(bench, [])
            avg = sum(bench_scores) / len(bench_scores) if bench_scores else 0.0
            avg_scores.append(avg)

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": data.get("model_name", "unknown"),
                    "data": scores,
                    "backgroundColor": "rgba(52, 152, 219, 0.2)",
                    "borderColor": "rgba(52, 152, 219, 1)",
                    "borderWidth": 2,
                },
                {
                    "label": "Average",
                    "data": avg_scores,
                    "backgroundColor": "rgba(149, 165, 166, 0.2)",
                    "borderColor": "rgba(149, 165, 166, 1)",
                    "borderWidth": 2,
                    "borderDash": [5, 5],
                },
            ],
        }

    def save_chart(self, chart_data: dict[str, Any], save_path: str) -> Path:
        """Save radar chart as PNG using matplotlib (optional dependency).

        Falls back to ASCII art if matplotlib is not available.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np

            angles = np.linspace(0, 2 * np.pi, len(chart_data["labels"]), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

            for dataset in chart_data["datasets"]:
                data = dataset["data"] + [dataset["data"][0]]
                ax.plot(angles, data, linewidth=2, label=dataset["label"],
                        color=dataset["borderColor"], linestyle="-" if "borderDash" not in dataset else "--")
                ax.fill(angles, data, color=dataset["backgroundColor"])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(chart_data["labels"], fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title("FairBench Radar Chart", fontsize=14, pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

            path = Path(save_path)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return path

        except ImportError:
            # Fallback to ASCII art
            return self._save_ascii_chart(chart_data, save_path)

    def _save_ascii_chart(self, chart_data: dict[str, Any], save_path: str) -> Path:
        """Fallback ASCII art radar chart."""
        lines: list[str] = []
        lines.append("FairBench Radar Chart (ASCII)")
        lines.append("=" * 50)

        for i, label in enumerate(chart_data["labels"]):
            lines.append(f"\n{label}:")
            for dataset in chart_data["datasets"]:
                score = dataset["data"][i]
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {dataset['label']}: [{bar}] {score:.2f}")

        path = Path(save_path)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
