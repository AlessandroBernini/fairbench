"""Reporting and publishing for FairBench."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from fairbench.config import load_config


class Reporter:
    """Handles result reporting and leaderboard publishing."""

    def __init__(self, config=None):
        self.config = config or load_config()

    def format_console_output(
        self,
        run_id: str,
        model_name: str,
        category: str,
        results: dict[str, Any],
    ) -> str:
        """Format benchmark results for console display."""
        lines = [
            f"\n{'=' * 60}",
            f"FairBench Result: {run_id}",
            f"Model: {model_name}",
            f"Category: {category}",
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"{'=' * 60}",
        ]

        if results.get("quality_results"):
            lines.append("\nQuality Results:")
            lines.append("-" * 40)
            for qr in results["quality_results"]:
                score = qr["score"]
                total = qr["items_total"]
                correct = qr["items_correct"]
                lines.append(f"  {qr['benchmark_name']}: {score:.4f} ({correct}/{total})")

        if results.get("speed_result"):
            sr = results["speed_result"]
            lines.append("\nSpeed Results:")
            lines.append("-" * 40)
            lines.append(f"  TTFT: {sr['ttft_avg']:.2f} ms")
            lines.append(f"  TPOT: {sr['tpot_avg']:.4f} tokens/s")
            lines.append(f"  Throughput: {sr['throughput_avg']:.2f} tokens/s")

        if results.get("memory_result"):
            mr = results["memory_result"]
            lines.append("\nMemory Results:")
            lines.append("-" * 40)
            lines.append(f"  VRAM Peak: {mr['vram_peak_mb']:.0f} MB")
            lines.append(f"  RAM Peak: {mr['ram_peak_mb']:.0f} MB")

        if results.get("swebench_results"):
            lines.append("\nSWE-bench Results:")
            lines.append("-" * 40)
            resolved = sum(1 for r in results["swebench_results"] if r.get("resolved"))
            total = len(results["swebench_results"])
            lines.append(
                f"  Resolved: {resolved}/{total} ({resolved / total * 100:.1f}%)"
                if total > 0
                else "  No results"
            )

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    def publish_result(self, run_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """Publish result to the FairBench leaderboard API."""
        if not self.config.publish:
            return None

        payload = {
            "run_id": run_id,
            "model_name": result["model_name"],
            "model_path": result["model_path"],
            "quantization": result["quantization"],
            "n_ctx": result["n_ctx"],
            "n_gpu_layers": result["n_gpu_layers"],
            "n_threads": result["n_threads"],
            "timestamp": result["timestamp"],
            "category": result["category"],
            "quality_scores": {
                qr["benchmark_name"]: qr["score"] for qr in result.get("quality_results", [])
            },
            "speed": {
                "ttft_ms": result.get("speed_result", {}).get("ttft_avg", 0),
                "tpot_tokens_per_sec": result.get("speed_result", {}).get("tpot_avg", 0),
                "throughput_tokens_per_sec": result.get("speed_result", {}).get(
                    "throughput_avg", 0
                ),
            },
            "memory": {
                "vram_peak_mb": result.get("memory_result", {}).get("vram_peak_mb", 0),
                "ram_peak_mb": result.get("memory_result", {}).get("ram_peak_mb", 0),
            },
            "swebench": {
                "resolved": sum(
                    1 for r in result.get("swebench_results", []) if r.get("resolved")
                ),
                "total": len(result.get("swebench_results", [])),
            },
        }

        try:
            response = requests.post(
                f"{self.config.api_url}/results",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            resp_data = response.json()
            assert isinstance(resp_data, dict)
            return resp_data
        except requests.RequestException as e:
            print(f"Publish failed: {e}")
            return None

    def get_leaderboard(self, limit: int = 50) -> list[dict]:
        """Fetch latest results from the leaderboard."""
        try:
            response = requests.get(
                f"{self.config.api_url}/results",
                params={"limit": limit},
                timeout=30,
            )
            response.raise_for_status()
            resp_data = response.json()
            assert isinstance(resp_data, list)
            return resp_data  # type: ignore[return-value]
        except requests.RequestException as e:
            print(f"Leaderboard fetch failed: {e}")
            return []

    def save_local_report(self, run_id: str, result: dict[str, Any]) -> Path:
        """Save a local JSON report."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / f"{run_id}_report.json"
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return filepath
