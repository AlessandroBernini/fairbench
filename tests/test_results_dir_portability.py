"""Regression coverage for portable benchmark result paths."""

from pathlib import Path

from fairbench.types import BenchmarkConfig


def test_results_dir_uses_current_home(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert BenchmarkConfig().results_dir == str(tmp_path / ".fairbench" / "results")


def test_results_dir_explicit_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert BenchmarkConfig(results_dir="/custom/results").results_dir == "/custom/results"


def test_results_dir_tracks_home_between_instances(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "first")
    first = BenchmarkConfig()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "second")
    second = BenchmarkConfig()
    assert first.results_dir == str(tmp_path / "first" / ".fairbench" / "results")
    assert second.results_dir == str(tmp_path / "second" / ".fairbench" / "results")
