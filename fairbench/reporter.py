"""Reporter for FairBench benchmarks."""

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests

from fairbench.config import Config


class Reporter:
    """Generate and manage benchmark reports."""

    def __init__(self, results_dir: Optional[Path] = None):
        config = Config()
        self.results_dir = results_dir or config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = config.api_url.rstrip('/')
        self.publish_default = config.publish_default

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
        """Generate a benchmark report and optionally publish it."""
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

        # Save locally
        filename = f"{timestamp}_{model_name}_{quantization}_{report_id}.json"
        filepath = self.results_dir / filename
        filepath.write_text(json.dumps(report, indent=2))

        # Publish if requested
        if publish:
            self._publish(report)

        return filepath

    def _publish(self, report: dict) -> bool:
        """Send report to the Vercel API."""
        try:
            resp = requests.post(
                f"{self.api_url}/report",
                json=report,
                timeout=30,
            )
            if resp.status_code == 201:
                return True
            else:
                print(f"Warning: publish failed ({resp.status_code}): {resp.text}")
                return False
        except requests.RequestException as e:
            print(f"Warning: publish failed ({e})")
            return False

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

    def get_leaderboard(self, model: Optional[str] = None, gpu: Optional[str] = None) -> list[dict]:
        """Fetch leaderboard data from the API."""
        try:
            params = {}
            if model:
                params["model"] = model
            if gpu:
                params["gpu"] = gpu
            resp = requests.get(
                f"{self.api_url}/leaderboard",
                params=params,
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("reports", [])
            return []
        except requests.RequestException:
            return []

    def get_model_stats(self) -> list[dict]:
        """Fetch model statistics from the API."""
        try:
            resp = requests.get(
                f"{self.api_url}/models",
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("models", [])
            return []
        except requests.RequestException:
            return []
