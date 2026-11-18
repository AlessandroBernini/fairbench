"""CLI coverage tests for run/docker/plugin commands (F6, part 2)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fairbench.cli import cli
from fairbench.config import BenchmarkConfig
from fairbench.types import BenchmarkCategory


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> BenchmarkConfig:
    config = BenchmarkConfig(results_dir=str(tmp_path / "results"))
    with patch("fairbench.cli.load_config", return_value=config):
        yield config


@pytest.fixture
def model_file(tmp_path: Path) -> str:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    return str(f)


# ---------------------------------------------------------------- run command

class TestRunCommand:
    def _engine_patch(self):
        return patch("fairbench.cli.BenchmarkEngine", return_value=MagicMock())

    def test_all_category_runs_three_benchmarks(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with self._engine_patch() as cls:
            result = runner.invoke(cli, ["run", model_file])
        assert result.exit_code == 0, result.output
        cats = [c.kwargs["category"] for c in cls.return_value.run_benchmark.call_args_list]
        assert cats == [
            BenchmarkCategory.QUALITY,
            BenchmarkCategory.SPEED,
            BenchmarkCategory.MEMORY,
        ]

    def test_single_category(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with self._engine_patch() as cls:
            result = runner.invoke(cli, ["run", model_file, "-k", "speed"])
        assert result.exit_code == 0, result.output
        assert cls.return_value.run_benchmark.call_count == 1
        assert (
            cls.return_value.run_benchmark.call_args.kwargs["category"]
            is BenchmarkCategory.SPEED
        )

    def test_swebench_category_overrides_split(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with self._engine_patch() as cls:
            result = runner.invoke(cli, ["run", model_file, "-k", "swebench", "-s", "lite"])
        assert result.exit_code == 0, result.output
        engine = cls.return_value
        assert (
            engine.run_benchmark.call_args.kwargs["category"] is BenchmarkCategory.SWE_BENCH
        )
        assert cfg.swebench_split == "lite"

    def test_publish_override(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with self._engine_patch():
            assert cfg.publish is True
            result = runner.invoke(cli, ["run", model_file, "-k", "speed", "--no-publish"])
        assert result.exit_code == 0, result.output
        assert cfg.publish is False

    def test_verbose_and_log_file(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str, tmp_path: Path
    ) -> None:
        log = tmp_path / "run.json"
        with self._engine_patch():
            result = runner.invoke(cli, ["run", model_file, "-k", "speed", "-vv", "--log-file", str(log)])
        assert result.exit_code == 0, result.output

    def test_tasks_forwarded(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with self._engine_patch() as cls:
            runner.invoke(cli, ["run", model_file, "-k", "quality", "-t", "mmlu", "-t", "gsm8k"])
        assert cls.return_value.run_benchmark.call_args.kwargs["tasks"] == ["mmlu", "gsm8k"]


# ---------------------------------------------------------------- docker mode

DOCKER_TARGETS = [
    "fairbench.docker.verify_docker_setup",
    "fairbench.docker.build_docker_image",
    "fairbench.docker.detect_gpu_support",
    "fairbench.docker.run_in_docker",
]


class TestDockerMode:
    def test_docker_unavailable_exits_1(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with patch(DOCKER_TARGETS[0], return_value={"docker_available": "No"}):
            result = runner.invoke(cli, ["run", model_file, "--docker"])
        assert result.exit_code == 1
        assert "Docker is not installed" in result.output

    def test_build_failure_exits_1(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with (
            patch(DOCKER_TARGETS[0], return_value={"docker_available": "Yes"}),
            patch(DOCKER_TARGETS[1], return_value=None),
        ):
            result = runner.invoke(cli, ["run", model_file, "--docker"])
        assert result.exit_code == 1
        assert "Failed to build Docker image" in result.output

    def test_success_propagates_container_exit_code(
        self, runner: CliRunner, cfg: BenchmarkConfig, model_file: str
    ) -> None:
        with (
            patch(DOCKER_TARGETS[0], return_value={"docker_available": "Yes"}),
            patch(DOCKER_TARGETS[1], return_value="fairbench:latest"),
            patch(DOCKER_TARGETS[2], return_value=False),
            patch(DOCKER_TARGETS[3], return_value=7) as run_in,
        ):
            result = runner.invoke(cli, ["run", model_file, "--docker", "--ci"])
        assert result.exit_code == 7
        kwargs = run_in.call_args.kwargs
        assert kwargs["image_name"] == "fairbench:latest"
        assert kwargs["ci_mode"] is True


# ---------------------------------------------------------------- plugins

def _plugin(name: str = "mmlu") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, version="1.0.0", category="quality", description="MMLU multi-task QA"
    )


class TestPluginCommands:
    def test_list_empty(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.plugins.base.PluginRegistry.list_plugins", return_value=[]):
            result = runner.invoke(cli, ["list-plugins"])
        assert "No plugins found" in result.output

    def test_list_with_plugins(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch(
            "fairbench.plugins.base.PluginRegistry.list_plugins", return_value=[_plugin()]
        ):
            result = runner.invoke(cli, ["list-plugins"])
        assert "mmlu" in result.output

    def test_info_found(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        p = _plugin()
        p.get_metadata = lambda: {"tasks": "57"}  # type: ignore[attr-defined]
        with patch("fairbench.plugins.base.PluginRegistry.get_plugin", return_value=p):
            result = runner.invoke(cli, ["plugin", "info", "mmlu"])
        assert "Plugin: mmlu" in result.output and "tasks: 57" in result.output

    def test_info_not_found_lists_available(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with (
            patch("fairbench.plugins.base.PluginRegistry.get_plugin", return_value=None),
            patch(
                "fairbench.plugins.base.PluginRegistry.list_plugins", return_value=[_plugin()]
            ),
        ):
            result = runner.invoke(cli, ["plugin", "info", "zzz"])
        assert "not found" in result.output and "mmlu" in result.output

    def test_install_success(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        run = MagicMock(return_value=MagicMock(stdout="Successfully installed fairbench-x"))
        with (
            patch("subprocess.run", run),
            patch("fairbench.plugins.base.PluginRegistry.discover_file_plugins"),
        ):
            result = runner.invoke(cli, ["plugin", "install", "fairbench-x"])
        assert result.exit_code == 0, result.output
        assert "installed successfully" in result.output

    def test_install_failure_exits_1(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        err = subprocess.CalledProcessError(1, ["pip"], stderr="no matching distribution")
        with patch("subprocess.run", side_effect=err):
            result = runner.invoke(cli, ["plugin", "install", "nonexistent-pkg"])
        assert result.exit_code == 1
        assert "Failed to install" in result.output

    def test_uninstall_removes_file(
        self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path
    ) -> None:
        custom = tmp_path / ".fairbench" / "plugins" / "custom"
        custom.mkdir(parents=True)
        target = custom / "mybench.py"
        target.write_text("# plugin\n")
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(cli, ["plugin", "uninstall", "mybench"])
        assert "Uninstalled plugin: mybench" in result.output
        assert not target.exists()

    def test_uninstall_missing_dir(
        self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path
    ) -> None:
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(cli, ["plugin", "uninstall", "ghost"])
        assert "No custom plugins directory" in result.output

    def test_uninstall_missing_file_lists_available(
        self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path
    ) -> None:
        custom = tmp_path / ".fairbench" / "plugins" / "custom"
        custom.mkdir(parents=True)
        (custom / "other.py").write_text("#\n")
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(cli, ["plugin", "uninstall", "ghost"])
        assert "not found" in result.output and "other" in result.output
