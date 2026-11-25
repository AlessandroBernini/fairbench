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
            lines.append(f"  TPOT: {sr['tpot_avg']:.4f} ms/token")
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

    def _resolve_hardware(self, result: dict[str, Any]) -> dict[str, Any]:
        """Map a benchmark result to the hardware payload expected by POST /api/report.

        Supports both the ``hardware_snapshot`` dict produced by the engine
        (snake_case keys) and flat legacy keys on the result itself.
        """
        hw = result.get("hardware_snapshot") or result.get("hardware") or {}
        if not isinstance(hw, dict):
            hw = {}

        cpu_model = str(hw.get("cpu_model") or result.get("cpu_model") or "")
        cpu_cores = int(hw.get("cpu_cores") or result.get("cpu_cores") or 0)
        cpu_threads = int(hw.get("cpu_threads") or result.get("cpu_threads") or 0)
        ram_gb = float(hw.get("total_ram_gb") or hw.get("ram_gb") or result.get("ram_gb") or result.get("total_ram_gb") or 0.0)
        gpu_type = str(hw.get("gpu_type") or result.get("gpu_type") or "none")
        gpu_model = str(hw.get("gpu_model") or result.get("gpu_model") or gpu_type)
        gpu_vram_gb = float(hw.get("gpu_vram_gb") or result.get("gpu_vram_gb") or 0.0)
        os_name = str(hw.get("os_platform") or hw.get("os") or result.get("os") or "")

        return {
            "cpu_model": cpu_model,
            "cpu_cores": cpu_cores,
            "cpu_threads": cpu_threads,
            "ram_gb": ram_gb,
            "gpu_type": gpu_type,
            "gpu_model": gpu_model,
            "gpu_vram_gb": gpu_vram_gb,
            "os": os_name,
        }

    def publish_result(self, run_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        """Publish result to the FairBench leaderboard API."""
        if not self.config.publish:
            return None

        # Map MB values to GB for the API schema
        memory_result = result.get("memory_result") or {}
        vram_peak_gb = float(memory_result.get("vram_peak_mb") or 0) / 1024
        ram_peak_gb = float(memory_result.get("ram_peak_mb") or 0) / 1024

        payload = {
            "run_id": run_id,
            "model_name": result["model_name"],
            "quantization": result["quantization"],
            "n_ctx": result["n_ctx"],
            "n_gpu_layers": result["n_gpu_layers"],
            "n_threads": result["n_threads"],
            "timestamp": result["timestamp"],
            "category": result["category"],
            "hardware": self._resolve_hardware(result),
            "quality_scores": {
                qr["benchmark_name"]: qr["score"] for qr in result.get("quality_results", [])
            },
            "speed": {
                "ttft_ms": (result.get("speed_result") or {}).get("ttft_avg") or 0,
                "tpot_ms": (result.get("speed_result") or {}).get("tpot_avg") or 0,
                "throughput": (result.get("speed_result") or {}).get("throughput_avg") or 0,
            },
            "memory": {
                "vram_peak_gb": vram_peak_gb,
                "cpu_ram_peak_gb": ram_peak_gb,
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
                f"{self.config.api_url}/report",
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
                f"{self.config.api_url}/leaderboard",
                params={"limit": limit},
                timeout=30,
            )
            response.raise_for_status()
            resp_data = response.json()
            reports = resp_data.get("reports", []) if isinstance(resp_data, dict) else resp_data
            assert isinstance(reports, list)
            return reports  # type: ignore[return-value]
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
