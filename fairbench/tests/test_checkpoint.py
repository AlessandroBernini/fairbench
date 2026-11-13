"""Tests for fairbench.checkpoint — checkpoint & resume mechanism."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from fairbench.checkpoint import (
    BenchmarkCheckpoint,
    cleanup_old_checkpoints,
    clear_checkpoint,
    get_next_benchmark,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from fairbench.types import LiteMode


class TestBenchmarkCheckpoint:
    """Test BenchmarkCheckpoint serialization."""

    def test_to_dict(self):
        checkpoint = BenchmarkCheckpoint(
            model_path="/path/to/model.gguf",
            category="quality",
            completed_benchmarks=["mmlu", "hellaswag"],
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        data = checkpoint.to_dict()
        assert data["model_path"] == "/path/to/model.gguf"
        assert data["category"] == "quality"
        assert data["completed_benchmarks"] == ["mmlu", "hellaswag"]

    def test_from_dict(self):
        data = {
            "model_path": "/path/to/model.gguf",
            "category": "quality",
            "completed_benchmarks": ["mmlu"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "lite_mode": "auto",
        }
        checkpoint = BenchmarkCheckpoint.from_dict(data)
        assert checkpoint.model_path == "/path/to/model.gguf"
        assert checkpoint.completed_benchmarks == ["mmlu"]

    def test_from_dict_with_resumed_from(self):
        data = {
            "model_path": "/path/to/model.gguf",
            "category": "quality",
            "completed_benchmarks": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "resumed_from": "run_123",
            "lite_mode": "auto",
        }
        checkpoint = BenchmarkCheckpoint.from_dict(data)
        assert checkpoint.resumed_from == "run_123"


class TestCheckpointSaveLoad:
    """Test checkpoint save and load operations."""

    @pytest.fixture
    def mock_checkpoint_dir(self, tmp_path):
        """Create a temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    def test_save_and_load_checkpoint(self, tmp_path, mock_checkpoint_dir):
        """Test saving and loading a checkpoint."""
        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=mock_checkpoint_dir):
            run_id = "test_run_123"
            progress = {
                "model_path": "/path/to/model.gguf",
                "category": "quality",
                "completed_benchmarks": ["mmlu"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

            # Save checkpoint
            path = save_checkpoint(run_id, progress)
            assert path.exists()

            # Load checkpoint
            checkpoint = load_checkpoint(run_id)
            assert checkpoint is not None
            assert checkpoint.model_path == "/path/to/model.gguf"
            assert checkpoint.completed_benchmarks == ["mmlu"]

    def test_load_nonexistent_checkpoint(self, mock_checkpoint_dir):
        """Test loading a checkpoint that doesn't exist."""
        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=mock_checkpoint_dir):
            checkpoint = load_checkpoint("nonexistent_run")
            assert checkpoint is None

    def test_clear_checkpoint(self, tmp_path, mock_checkpoint_dir):
        """Test clearing a checkpoint."""
        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=mock_checkpoint_dir):
            run_id = "test_run_456"
            progress = {
                "model_path": "/path/to/model.gguf",
                "category": "quality",
                "completed_benchmarks": [],
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(run_id, progress)

            # Clear checkpoint
            result = clear_checkpoint(run_id)
            assert result is True

            # Verify it's gone
            checkpoint = load_checkpoint(run_id)
            assert checkpoint is None

    def test_clear_nonexistent_checkpoint(self, mock_checkpoint_dir):
        """Test clearing a checkpoint that doesn't exist."""
        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=mock_checkpoint_dir):
            result = clear_checkpoint("nonexistent_run")
            assert result is False


class TestListCheckpoints:
    """Test listing checkpoints."""

    def test_list_empty_directory(self, tmp_path):
        """Test listing checkpoints when directory is empty."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=checkpoint_dir):
            checkpoints = list_checkpoints()
            assert checkpoints == []

    def test_list_with_checkpoints(self, tmp_path):
        """Test listing checkpoints with existing files."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create mock checkpoint files
        for i in range(3):
            cp_path = checkpoint_dir / f"run_{i}.json"
            data = {
                "run_id": f"run_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "progress": {
                    "model_path": "/path/to/model.gguf",
                    "category": "quality",
                    "completed_benchmarks": [],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            with open(cp_path, "w") as f:
                json.dump(data, f)

        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=checkpoint_dir):
            checkpoints = list_checkpoints()
            assert len(checkpoints) == 3
            assert all("run_id" in cp for cp in checkpoints)


class TestCleanupOldCheckpoints:
    """Test automatic cleanup of old checkpoints."""

    def test_cleanup_removes_old_checkpoints(self, tmp_path):
        """Test that old checkpoints are removed."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create an old checkpoint (48 hours ago)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        old_cp = checkpoint_dir / "old_run.json"
        with open(old_cp, "w") as f:
            json.dump({
                "run_id": "old_run",
                "timestamp": old_time,
                "progress": {},
            }, f)

        # Create a recent checkpoint (1 hour ago)
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        recent_cp = checkpoint_dir / "recent_run.json"
        with open(recent_cp, "w") as f:
            json.dump({
                "run_id": "recent_run",
                "timestamp": recent_time,
                "progress": {},
            }, f)

        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=checkpoint_dir):
            removed = cleanup_old_checkpoints(max_age_hours=24)
            assert removed == 1
            assert not old_cp.exists()
            assert recent_cp.exists()

    def test_cleanup_nothing_to_remove(self, tmp_path):
        """Test cleanup when no checkpoints are old."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a recent checkpoint
        recent_time = datetime.now(timezone.utc).isoformat()
        recent_cp = checkpoint_dir / "recent_run.json"
        with open(recent_cp, "w") as f:
            json.dump({
                "run_id": "recent_run",
                "timestamp": recent_time,
                "progress": {},
            }, f)

        with patch("fairbench.checkpoint._get_checkpoint_dir", return_value=checkpoint_dir):
            removed = cleanup_old_checkpoints(max_age_hours=24)
            assert removed == 0
            assert recent_cp.exists()


class TestGetNextBenchmark:
    """Test benchmark selection logic."""

    def test_get_next_benchmark_finds_first_missing(self):
        """Test finding the first missing benchmark."""
        benchmarks = ["mmlu", "hellaswag", "gsm8k"]
        completed = ["mmlu"]

        next_bench = get_next_benchmark(benchmarks, completed, LiteMode.AUTO)
        assert next_bench == "hellaswag"

    def test_get_next_benchmark_returns_none_when_all_done(self):
        """Test returning None when all benchmarks are completed."""
        benchmarks = ["mmlu", "hellaswag"]
        completed = ["mmlu", "hellaswag"]

        next_bench = get_next_benchmark(benchmarks, completed, LiteMode.AUTO)
        assert next_bench is None

    def test_get_next_benchmark_skips_non_lite_when_enabled(self):
        """Test skipping non-lite benchmarks when lite mode is enabled."""
        benchmarks = ["mmlu", "hellaswag", "humaneval"]  # humaneval is not lite
        completed = ["mmlu"]

        next_bench = get_next_benchmark(benchmarks, completed, LiteMode.ENABLED)
        assert next_bench == "hellaswag"

    def test_get_next_benchmark_runs_all_when_disabled(self):
        """Test running all benchmarks when lite mode is disabled."""
        benchmarks = ["mmlu", "hellaswag", "humaneval"]
        completed = ["mmlu"]

        next_bench = get_next_benchmark(benchmarks, completed, LiteMode.DISABLED)
        assert next_bench == "hellaswag"
