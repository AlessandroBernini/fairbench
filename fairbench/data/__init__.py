"""Data management for FairBench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fairbench.config import ensure_results_dir


class DataManager:
    """Manages local data storage for FairBench."""

    def __init__(self, results_dir: str | None = None):
        self.results_dir = Path(ensure_results_dir(results_dir))
        self.cache_dir = self.results_dir / "cache"
        self.swebench_dir = self.results_dir / "swebench"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.swebench_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, run_id: str, data: dict[str, Any], category: str = "run") -> Path:
        """Save benchmark result as JSON."""
        filename = f"{run_id}.json"
        filepath = self.results_dir / category / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def load_result(self, run_id: str, category: str = "run") -> dict[str, Any] | None:
        """Load benchmark result from JSON."""
        filepath = self.results_dir / category / f"{run_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def list_results(self, category: str = "run") -> list[str]:
        """List all saved result IDs."""
        category_dir = self.results_dir / category
        if not category_dir.exists():
            return []
        return [f.stem for f in category_dir.glob("*.json")]

    def save_swebench_instance(self, instance_id: str, data: dict[str, Any]) -> Path:
        """Save SWE-bench instance data."""
        filepath = self.swebench_dir / f"{instance_id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def load_swebench_instance(self, instance_id: str) -> dict[str, Any] | None:
        """Load SWE-bench instance data."""
        filepath = self.swebench_dir / f"{instance_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def get_model_hash(self, model_path: str) -> str:
        """Get SHA256 hash of model file for caching."""
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_cached_result(self, model_hash: str, benchmark: str) -> dict[str, Any] | None:
        """Check if cached result exists for this model+benchmark."""
        cache_key = f"{model_hash}_{benchmark}"
        filepath = self.cache_dir / f"{cache_key}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def save_cached_result(self, model_hash: str, benchmark: str, data: dict[str, Any]) -> None:
        """Save result to cache."""
        cache_key = f"{model_hash}_{benchmark}"
        filepath = self.cache_dir / f"{cache_key}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def clear_cache(self) -> int:
        """Clear cache directory. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
