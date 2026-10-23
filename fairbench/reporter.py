"""Reporter for FairBench benchmarks."""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional


class Reporter:
    """Generate and manage benchmark reports."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        model_name: str,
        quantization: str,
        hardware: dict,
        quality: dict,
        speed: dict,
        memory: dict,
        publish: bool = True,
    ) -> Path:
        """Generate a benchmark report."""
        report_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        report = {
            "id": report_id,
            "timestamp": timestamp,
            "published": publish,
            "model": {
                "name": model_name,
                "quantization": quantization,
            },
            "hardware": hardware,
            "quality": quality,
            "speed": speed,
            "memory": memory,
        }

        filename = f"{timestamp}_{model_name}_{quantization}_{report_id}.json"
        filepath = self.results_dir / filename
        filepath.write_text(json.dumps(report, indent=2))

        return filepath

    def get_latest(self) -> Optional[dict]:
        """Get the latest report."""
        files = sorted(self.results_dir.glob("*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())

    def list_reports(self, limit: int = 10) -> list[dict]:
        """List recent reports."""
        files = sorted(self.results_dir.glob("*.json"), reverse=True)[:limit]
        reports = []
        for f in files:
            report = json.loads(f.read_text())
            reports.append({
                "id": report["id"],
                "timestamp": report["timestamp"],
                "model": report["model"]["name"],
                "published": report.get("published", True),
            })
        return reports
