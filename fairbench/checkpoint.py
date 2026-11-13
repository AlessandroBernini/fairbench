"""Checkpoint & resume mechanism for long benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fairbench.config import CONFIG_PATH
from fairbench.types import LiteMode


@dataclass
class BenchmarkCheckpoint:
    """Stores progress for a long benchmark run."""

    model_path: str
    category: str
    completed_benchmarks: list[str]
    started_at: str
    resumed_from: str | None = None
    hardware_profile_hash: str | None = None
    lite_mode: str = "auto"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkCheckpoint:
        """Create from dictionary."""
        return cls(
            model_path=data["model_path"],
            category=data["category"],
            completed_benchmarks=data["completed_benchmarks"],
            started_at=data["started_at"],
            resumed_from=data.get("resumed_from"),
            hardware_profile_hash=data.get("hardware_profile_hash"),
            lite_mode=data.get("lite_mode", "auto"),
        )


def _get_checkpoint_dir() -> Path:
    """Get the directory for checkpoint files."""
    checkpoint_dir = CONFIG_PATH.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _get_checkpoint_path(run_id: str) -> Path:
    """Get the path for a checkpoint file."""
    return _get_checkpoint_dir() / f"{run_id}.json"


def save_checkpoint(run_id: str, progress: dict) -> Path:
    """Save a checkpoint to disk.

    Args:
        run_id: Unique identifier for the benchmark run.
        progress: Dictionary containing progress data.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = _get_checkpoint_path(run_id)
    data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "progress": progress,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)
    return checkpoint_path


def load_checkpoint(run_id: str) -> BenchmarkCheckpoint | None:
    """Load a checkpoint from disk.

    Args:
        run_id: Unique identifier for the benchmark run.

    Returns:
        BenchmarkCheckpoint if found, None otherwise.
    """
    checkpoint_path = _get_checkpoint_path(run_id)
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        progress = data.get("progress", {})
        return BenchmarkCheckpoint.from_dict(progress)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_checkpoint(run_id: str) -> bool:
    """Clear a checkpoint file.

    Args:
        run_id: Unique identifier for the benchmark run.

    Returns:
        True if the checkpoint was deleted, False if it didn't exist.
    """
    checkpoint_path = _get_checkpoint_path(run_id)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    return False


def list_checkpoints() -> list[dict]:
    """List all active checkpoints.

    Returns:
        List of checkpoint info dictionaries.
    """
    checkpoint_dir = _get_checkpoint_dir()
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            checkpoints.append({
                "run_id": data.get("run_id", path.stem),
                "timestamp": data.get("timestamp", "unknown"),
                "progress": data.get("progress", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return checkpoints


def get_next_benchmark(
    all_benchmarks: list[str],
    completed: list[str],
    lite_mode: LiteMode,
) -> str | None:
    """Return next benchmark to run, or None if all done.

    Args:
        all_benchmarks: List of available benchmarks.
        completed: List of completed benchmark names.
        lite_mode: Lite mode setting.

    Returns:
        Next benchmark name, or None if all completed.
    """
    completed_set = set(completed)
    for benchmark in all_benchmarks:
        if benchmark not in completed_set:
            # Skip non-lite benchmarks if lite_mode is enabled
            from fairbench.quality import get_lite_benchmarks

            if lite_mode == LiteMode.ENABLED:
                lite_benchmarks = get_lite_benchmarks()
                if benchmark not in lite_benchmarks:
                    continue
            return benchmark
    return None


def cleanup_old_checkpoints(max_age_hours: float = 24) -> int:
    """Remove checkpoints older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before cleanup.

    Returns:
        Number of checkpoints removed.
    """
    checkpoint_dir = _get_checkpoint_dir()
    if not checkpoint_dir.exists():
        return 0

    removed = 0
    for path in checkpoint_dir.glob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            timestamp = data.get("timestamp", "")
            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                if age_hours > max_age_hours:
                    path.unlink()
                    removed += 1
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return removed
