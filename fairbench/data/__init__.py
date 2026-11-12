"""Data management for FairBench."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from fairbench.config import ensure_results_dir
from fairbench.models import _parse_gguf_header


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
            data = json.load(f)
        assert isinstance(data, dict)
        return data

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
            data = json.load(f)
        assert isinstance(data, dict)
        return data

    def get_model_hash(self, model_path: str) -> str:
        """Get SHA256 hash of model file for caching."""
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_model_cache_key(self, model_path: str, benchmark: str) -> tuple[str, str]:
        """Return (file_hash, metadata_hash) tuple for smart caching.

        Uses dual-key scheme: file SHA256 + GGUF metadata hash.
        If GGUF parsing fails, falls back to file-only key.
        """
        file_hash = self.get_model_hash(model_path)

        try:
            gguf_meta = _parse_gguf_header(Path(model_path))
            meta_str = str(
                sorted(
                    k
                    for k in gguf_meta
                    if k not in ("file_size_bytes",)
                )
            )
            meta_hash = hashlib.md5(meta_str.encode()).hexdigest()
            return (file_hash, meta_hash)
        except (OSError, ValueError):
            return (file_hash, "")

    def get_cached_result(self, model_hash: str, benchmark: str) -> Any:
        """Check if cached result exists for this model+benchmark.

        Returns the cached data (dict, list, or scalar) or None if not found.
        """
        cache_key = f"{model_hash}_{benchmark}"
        filepath = self.cache_dir / f"{cache_key}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def save_cached_result(
        self,
        model_hash: str,
        benchmark: str,
        data: Any,
        ttl_days: int | None = None,
    ) -> None:
        """Save result to cache with optional TTL.

        Args:
            model_hash: Model file hash.
            benchmark: Benchmark name.
            data: Data to cache.
            ttl_days: Time-to-live in days. None = indefinite.
        """
        cache_key = f"{model_hash}_{benchmark}"
        filepath = self.cache_dir / f"{cache_key}.json"

        cache_meta: dict[str, Any] = {}
        if ttl_days is not None:
            cache_meta["cached_at"] = time.time()
            cache_meta["ttl_days"] = ttl_days

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if cache_meta:
            meta_path = self.cache_dir / f"{cache_key}.meta.json"
            with open(meta_path, "w") as f:
                json.dump(cache_meta, f)

    def is_cache_valid(self, model_hash: str, benchmark: str) -> bool:
        """Check if cached result exists and is still valid."""
        cache_key = f"{model_hash}_{benchmark}"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"
        if not meta_path.exists():
            return True  # No TTL = indefinite
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cached_at = meta.get("cached_at", 0)
            ttl_days = meta.get("ttl_days")
            if ttl_days is None:
                return True
            age_days = (time.time() - cached_at) / 86400
            return age_days < ttl_days
        except (json.JSONDecodeError, OSError, KeyError):
            return True

    def clear_cache(self) -> int:
        """Clear cache directory. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
