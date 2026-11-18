"""Deep coverage tests for fairbench.backend (F6).

Exercises the llama-cpp-python code paths with injected fake modules —
no real model or native library required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from fairbench.backend import (
    _benchmark_with_llama_bench,
    benchmark_memory,
    benchmark_speed,
)


def _fake_llama_module(llm_instance: MagicMock) -> dict[str, ModuleType]:
    mod = ModuleType("llama_cpp")
    mod.Llama = MagicMock(return_value=llm_instance)  # type: ignore[attr-defined]
    return {"llama_cpp": mod}


def _make_llm(chunks: int = 5) -> MagicMock:
    llm = MagicMock()
    llm.create_completion.return_value = [{"choices": [{"text": "x"}]} for _ in range(chunks)]
    return llm


class TestBenchmarkSpeedLlamaCpp:
    def test_happy_path(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        llm = _make_llm()
        with patch.dict("sys.modules", _fake_llama_module(llm)):
            result = benchmark_speed(str(model), runs=2)
        assert result.runs == 2
        assert result.ttft_avg > 0
        assert result.throughput_avg >= 0
        assert llm.close.call_count == 2

    def test_run_failure_is_swallowed_per_run(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with patch.dict("sys.modules", {"llama_cpp": ModuleType("llama_cpp")}):
            import sys

            sys.modules["llama_cpp"].Llama = MagicMock(side_effect=OSError("boom"))  # type: ignore[attr-defined]
            result = benchmark_speed(str(model), runs=3)
        assert result.runs == 0
        assert result.ttft_avg == 0.0


class TestBenchmarkSpeedFallback:
    def test_llama_bench_output_captured(self) -> None:
        run = MagicMock(return_value=MagicMock(stdout="llama-bench raw output", returncode=0))
        with patch("fairbench.backend.subprocess.run", run):
            result = _benchmark_with_llama_bench("/x/m.gguf", "prompt words here", 2)
        assert result.metadata["raw_output"] == "llama-bench raw output"

    def test_llama_bench_missing_binary(self) -> None:
        with patch("fairbench.backend.subprocess.run", side_effect=FileNotFoundError()):
            result = _benchmark_with_llama_bench("/x/m.gguf", "p", 1)
        assert "error" in result.metadata

    def test_llama_bench_timeout(self) -> None:
        with patch(
            "fairbench.backend.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="llama-bench", timeout=300),
        ):
            result = _benchmark_with_llama_bench("/x/m.gguf", "p", 1)
        assert "error" in result.metadata

    def test_speed_dispatches_to_fallback_without_bindings(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with (
            patch.dict("sys.modules", {"llama_cpp": None}),
            patch("fairbench.backend._benchmark_with_llama_bench") as fb,
        ):
            benchmark_speed(str(model), runs=1)
        fb.assert_called_once()


class TestBenchmarkMemoryLlamaCpp:
    def test_ram_measured_no_gpu(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 2048)
        llm = _make_llm()
        with (
            patch.dict("sys.modules", _fake_llama_module(llm)),
            patch("fairbench.backend.os.path.exists", return_value=False),
        ):
            result = benchmark_memory(str(model))
        assert "error" not in result.metadata
        assert llm.create_completion.called
        assert llm.close.called

    def test_nvidia_smi_samples_parsed(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 2048)
        llm = _make_llm()
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="1000\n3000\n"))
        with (
            patch.dict("sys.modules", _fake_llama_module(llm)),
            patch("fairbench.backend.os.path.exists", return_value=True),
            patch("fairbench.backend.subprocess.run", run),
        ):
            result = benchmark_memory(str(model))
        assert result.vram_peak_mb == pytest.approx(2000.0)

    def test_error_path_returns_metadata(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with patch.dict("sys.modules", {"llama_cpp": ModuleType("llama_cpp")}):
            import sys

            sys.modules["llama_cpp"].Llama = MagicMock(side_effect=RuntimeError("cuda oom"))  # type: ignore[attr-defined]
            result = benchmark_memory(str(model))
        assert "cuda oom" in result.metadata.get("error", "")

    def test_memory_dispatches_without_bindings(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = benchmark_memory(str(model))
        assert result.metadata.get("reason") == "llama-cpp-python not installed"
