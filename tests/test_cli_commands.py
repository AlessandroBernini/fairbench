"""CLI command coverage tests via click CliRunner (F6).

All filesystem side effects are redirected to tmp dirs; engines and external
processes are mocked at the boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fairbench import cli as cli_module
from fairbench.cli import _estimate_model_size, _resolve_model_path, cli
from fairbench.config import BenchmarkConfig


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cfg(tmp_path: Path) -> BenchmarkConfig:
    config = BenchmarkConfig(results_dir=str(tmp_path / "results"))
    with patch("fairbench.cli.load_config", return_value=config):
        yield config


# ---------------------------------------------------------------- registration

class TestRegistration:
    def test_download_is_registered_in_group(self) -> None:
        """`fairbench download` is documented in help — it must be reachable."""
        assert "download" in cli.commands

    def test_all_documented_commands_present(self) -> None:
        for name in (
            "run",
            "verify",
            "quick-start",
            "show-paths",
            "list-benchmarks",
            "list-swebench",
            "config",
            "leaderboard",
            "results",
            "export",
            "report",
            "compare",
            "models",
            "plugin",
        ):
            assert name in cli.commands, f"missing command: {name}"


# ---------------------------------------------------------------- simple passthroughs

class TestPassthroughCommands:
    @pytest.mark.parametrize(
        ("args", "target"),
        [
            (["show-paths"], "fairbench.install.cmd_show_paths"),
            (["verify"], "fairbench.install.cmd_verify"),
            (["quick-start"], "fairbench.install.cmd_quick_start"),
        ],
    )
    def test_passthrough_invoked(self, runner: CliRunner, cfg: BenchmarkConfig, args: list, target: str) -> None:
        with patch(target) as fn:
            result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        fn.assert_called_once()

    def test_download_passthrough_forwards_args(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.install.cmd_download") as fn:
            result = runner.invoke(cli, ["download", "Qwen/Test", "-f", "m.gguf", "-d", "/tmp/x"])
        assert result.exit_code == 0, result.output
        fn.assert_called_once_with(model_id="Qwen/Test", filename="m.gguf", local_dir="/tmp/x")


# ---------------------------------------------------------------- init / config file commands

class TestInitCmd:
    def test_creates_missing_config(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        with (
            patch("fairbench.config.CONFIG_PATH", target),
            patch("fairbench.config.save_config") as save,
        ):
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "Config created" in result.output
        save.assert_called_once()

    def test_existing_config_warns(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        target.write_text("publish: true\n")
        with patch("fairbench.config.CONFIG_PATH", target):
            result = runner.invoke(cli, ["init"])
        assert "already exists" in result.output

    def test_wizard_flag_delegates_to_install_cmd_init(
        self, runner: CliRunner, cfg: BenchmarkConfig
    ) -> None:
        """`init --wizard` must run the interactive setup wizard."""
        with patch("fairbench.install.cmd_init") as wizard:
            result = runner.invoke(cli, ["init", "--wizard"])
        assert result.exit_code == 0, result.output
        wizard.assert_called_once()

    def test_default_init_never_prompts(
        self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """`init` without --wizard stays scriptable: no interactive delegation."""
        target = tmp_path / "cfg.yaml"
        with (
            patch("fairbench.config.CONFIG_PATH", target),
            patch("fairbench.config.save_config"),
            patch("fairbench.install.cmd_init") as wizard,
        ):
            result = runner.invoke(cli, ["init"], input="")
        assert result.exit_code == 0, result.output
        wizard.assert_not_called()


class TestCliSourceIntegrity:
    """Structural guards against silent registration bugs (F6 download / F8 init)."""

    @staticmethod
    def _registered_names() -> list[str]:
        tree = ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name)
                    and dec.func.value.id == "cli"
                    and dec.func.attr == "command"
                ):
                    first = dec.args[0] if dec.args else None
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        names.append(first.value)
                    else:
                        names.append(node.name)
        return names

    def test_no_duplicate_cli_command_registrations(self) -> None:
        names = self._registered_names()
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"duplicate @cli.command registrations (last silently wins): {dupes}"

    def test_every_registered_command_reachable_in_group(self) -> None:
        missing = [n for n in self._registered_names() if n not in cli.commands]
        assert not missing, f"@cli.command names absent from cli.commands: {missing}"


class TestValidateConfigCmd:
    def test_no_file_ok(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        with patch("fairbench.config.CONFIG_PATH", tmp_path / "none.yaml"):
            result = runner.invoke(cli, ["validate-config"])
        assert "No config file" in result.output

    def test_valid_file(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        target.write_text("publish: true\ndefault_n_ctx: 4096\n")
        with patch("fairbench.config.CONFIG_PATH", target):
            result = runner.invoke(cli, ["validate-config"])
        assert result.exit_code == 0
        assert "valid" in result.output

    def test_invalid_file_exits_1(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        target.write_text("default_n_ctx: not-an-int\n")
        with patch("fairbench.config.CONFIG_PATH", target):
            result = runner.invoke(cli, ["validate-config"])
        assert result.exit_code == 1
        assert "Config error" in result.output


class TestFixConfigCmd:
    def test_no_file(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        with patch("fairbench.config.CONFIG_PATH", tmp_path / "none.yaml"):
            result = runner.invoke(cli, ["fix-config"])
        assert "Nothing to fix" in result.output

    def test_coerces_and_prunes(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        target.write_text('publish: "false"\nbogus_key: 1\n')
        with (
            patch("fairbench.config.CONFIG_PATH", target),
            patch("fairbench.config.save_config") as save,
        ):
            result = runner.invoke(cli, ["fix-config"])
        assert result.exit_code == 0, result.output
        assert "Removing unknown keys" in result.output
        saved_cfg = save.call_args[0][0]
        assert saved_cfg.publish is False

    def test_unfixable_exits_1(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        target = tmp_path / "cfg.yaml"
        target.write_text("default_n_ctx: garbage\n")
        with patch("fairbench.config.CONFIG_PATH", target):
            result = runner.invoke(cli, ["fix-config"])
        assert result.exit_code == 1
        assert "Fix failed" in result.output


# ---------------------------------------------------------------- checkpoints

class TestCheckpointCommands:
    def test_list_empty(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.checkpoint.list_checkpoints", return_value=[]):
            result = runner.invoke(cli, ["checkpoints"])
        assert "No active checkpoints" in result.output

    def test_list_with_entries(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        cps = [{"run_id": "r1", "timestamp": "2026-11-18T00:00:00"}]
        with patch("fairbench.checkpoint.list_checkpoints", return_value=cps):
            result = runner.invoke(cli, ["checkpoints"])
        assert "r1" in result.output

    def test_cleanup(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.checkpoint.cleanup_old_checkpoints", return_value=3) as cleanup:
            result = runner.invoke(cli, ["checkpoints", "--cleanup"])
        assert "Cleaned up 3" in result.output
        cleanup.assert_called_once()

    @pytest.mark.parametrize(("cleared", "expected"), [(True, "cleared"), (False, "No checkpoint found")])
    def test_clear(self, runner: CliRunner, cfg: BenchmarkConfig, cleared: bool, expected: str) -> None:
        with patch("fairbench.checkpoint.clear_checkpoint", return_value=cleared):
            result = runner.invoke(cli, ["clear-checkpoint", "r1"])
        assert expected in result.output


# ---------------------------------------------------------------- config command

class TestConfigCommand:
    def test_show(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        result = runner.invoke(cli, ["config", "--show"])
        assert result.exit_code == 0
        assert "api_url" in result.output

    def test_set_publish_bool_coercion(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.save_config"):
            result = runner.invoke(cli, ["config", "--set", "publish=false"])
        assert "Set publish = False" in result.output
        assert cfg.publish is False

    def test_set_int_coercion(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.save_config"):
            result = runner.invoke(cli, ["config", "--set", "default_n_ctx=8192"])
        assert "Set default_n_ctx" in result.output
        assert cfg.default_n_ctx == 8192

    def test_set_bad_int(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.save_config"):
            result = runner.invoke(cli, ["config", "--set", "default_threads=nope"])
        assert "must be an integer" in result.output

    def test_set_unknown_key(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.save_config"):
            result = runner.invoke(cli, ["config", "--set", "nope=1"])
        assert "Unknown config key" in result.output

    def test_set_missing_equals(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.save_config"):
            result = runner.invoke(cli, ["config", "--set", "broken"])
        assert "Invalid key=value" in result.output


# ---------------------------------------------------------------- engine-backed commands

def _mock_engine() -> MagicMock:
    engine = MagicMock()
    return engine


class TestEngineCommands:
    def test_list_benchmarks(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.list_available_benchmarks.return_value = {
                "mmlu": {"name": "MMLU", "description": "Multi-task QA", "lite": True},
            }
            result = runner.invoke(cli, ["list-benchmarks"])
        assert "MMLU" in result.output and "[lite]" in result.output

    def test_list_swebench(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.get_swebench_instances.return_value = [
                {"instance_id": "repo__pr1", "repo": "repo", "code_language": "python"},
            ]
            result = runner.invoke(cli, ["list-swebench", "-s", "lite", "-l", "5"])
        assert "repo__pr1" in result.output

    def test_swebench_eval_no_instances(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.swebench_client.fetch_instances.return_value = []
            result = runner.invoke(cli, ["swebench-eval", "model.gguf"])
        assert "No instances available" in result.output

    def test_swebench_eval_with_results(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        stats = SimpleNamespace(
            total_instances=2, resolved=1, failed=1, errors=0, avg_test_score=50.0
        )
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            client = cls.return_value.swebench_client
            client.fetch_instances.return_value = [{"instance_id": "a"}, {"instance_id": "b"}]
            client.run_evaluation.return_value = []
            client.compute_stats.return_value = stats
            result = runner.invoke(cli, ["swebench-eval", "model.gguf", "-r", "repo"])
        assert "Found 2 instances" in result.output
        assert "Score: 50.0%" in result.output

    def test_leaderboard_empty(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.reporter.get_leaderboard.return_value = []
            result = runner.invoke(cli, ["leaderboard"])
        assert "No results available" in result.output

    def test_leaderboard_rows(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        rows = [
            {
                "model_name": "qwen-test",
                "quality_scores": {"arc": 60.0},
                "speed": {"throughput_tokens_per_sec": 12.5},
                "memory": {"vram_peak_mb": 4096},
                "swebench": {"resolved": 1, "total": 2},
            }
        ]
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.reporter.get_leaderboard.return_value = rows
            result = runner.invoke(cli, ["leaderboard"])
        assert "qwen-test" in result.output

    def test_results_empty(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.data.list_results.return_value = []
            result = runner.invoke(cli, ["results"])
        assert "No local results" in result.output

    def test_results_listed(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.cli.BenchmarkEngine", return_value=_mock_engine()) as cls:
            cls.return_value.data.list_results.return_value = ["run_1"]
            result = runner.invoke(cli, ["results"])
        assert "run_1" in result.output


# ---------------------------------------------------------------- radar / export / report

class TestReportCommands:
    def test_radar_json_output(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.radar.RadarEngine") as cls:
            cls.return_value.generate_radar.return_value = {"axes": ["q"], "values": [1]}
            result = runner.invoke(cli, ["radar", "run_1"])
        assert '"axes"' in result.output

    def test_radar_save(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.radar.RadarEngine"):
            result = runner.invoke(cli, ["radar", "run_1", "-s", "/tmp/r.png"])
        assert "Chart saved" in result.output

    def test_export_csv(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.export.ExportEngine") as cls:
            cls.return_value.export_csv.return_value = "run_id,score\nr1,42\n"
            result = runner.invoke(cli, ["export", "-f", "csv"])
        assert "run_id,score" in result.output

    def test_export_json(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.export.ExportEngine") as cls:
            cls.return_value.export_json.return_value = '{"runs": []}'
            result = runner.invoke(cli, ["export", "-f", "json"])
        assert '"runs"' in result.output

    def test_report(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.report.ReportEngine") as cls:
            cls.return_value.save_report.return_value = "/tmp/report.html"
            result = runner.invoke(cli, ["report", "run_1"])
        assert "Report generated" in result.output

    def test_compare_formats(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.compare.ComparisonEngine") as cls:
            res = cls.return_value.compare_runs.return_value
            res.to_json.return_value = "{}"
            res.to_csv.return_value = "a,b"
            res.to_console_table.return_value = "table"
            assert "{}" in runner.invoke(cli, ["compare", "-f", "json"]).output
            assert "a,b" in runner.invoke(cli, ["compare", "-f", "csv"]).output
            assert "table" in runner.invoke(cli, ["compare", "-s", "mmlu"]).output


# ---------------------------------------------------------------- models group

def _model_info(name: str = "qwen2.5-7b") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        architecture="qwen2",
        file_size_gb=4.2,
        quantization="Q4_K_M",
        path=Path("/models/qwen.gguf"),
        context_length=32768,
        quantization_version=1,
    )


class TestModelsCommands:
    def test_list_empty(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.list_models.return_value = []
            result = runner.invoke(cli, ["models", "list"])
        assert "No models found" in result.output

    def test_list_with_models(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.list_models.return_value = [_model_info()]
            result = runner.invoke(cli, ["models", "list"])
        assert "qwen2.5-7b" in result.output

    def test_search_hit_and_miss(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.search.return_value = [_model_info()]
            assert "matching" in runner.invoke(cli, ["models", "search", "qwen"]).output
            cat.return_value.search.return_value = []
            assert "No models matching" in runner.invoke(cli, ["models", "search", "zzz"]).output

    def test_info_found_and_missing(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.get_model.return_value = _model_info()
            assert "Architecture" in runner.invoke(cli, ["models", "info", "qwen2.5-7b"]).output
            cat.return_value.get_model.return_value = None
            assert "not found" in runner.invoke(cli, ["models", "info", "zzz"]).output

    def test_add_missing_file(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        result = runner.invoke(cli, ["models", "add", "/nope/x.gguf"])
        assert "File not found" in result.output

    def test_add_ok(self, runner: CliRunner, cfg: BenchmarkConfig, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.add_model.return_value = _model_info()
            result = runner.invoke(cli, ["models", "add", str(model)])
        assert "Added" in result.output and "scanned" in result.output

    def test_remove(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.remove_model.return_value = True
            assert "Removed" in runner.invoke(cli, ["models", "remove", "qwen"]).output
            cat.return_value.remove_model.return_value = False
            assert "not found" in runner.invoke(cli, ["models", "remove", "zzz"]).output

    def test_recommend_from_catalog(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        recs = {"gpu_layers": 20, "quant": "q4_k_m", "n_ctx": 4096, "threads": 4}
        with (
            patch("fairbench.models.ModelCatalog") as cat,
            patch("fairbench.hardware.detect_hardware"),
            patch("fairbench.hardware.generate_recommendation", return_value=recs),
        ):
            cat.return_value.get_model.return_value = _model_info()
            result = runner.invoke(cli, ["models", "recommend", "qwen2.5-7b"])
        assert "n_gpu_layers: 20" in result.output

    def test_recommend_size_unparsable(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.get_model.return_value = None
            result = runner.invoke(cli, ["models", "recommend", "/x/unknown-name.gguf"])
        assert "Cannot determine model size" in result.output


# ---------------------------------------------------------------- helpers

class TestHelpers:
    def test_resolve_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x")
        assert _resolve_model_path(str(f)) == str(f)

    def test_resolve_catalog_hit(self) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.get_model.return_value = SimpleNamespace(path=Path("/models/x.gguf"))
            assert _resolve_model_path("qwen2.5-7b") == "/models/x.gguf"

    def test_resolve_miss_passthrough(self) -> None:
        with patch("fairbench.models.ModelCatalog") as cat:
            cat.return_value.get_model.return_value = None
            assert _resolve_model_path("unknown-name") == "unknown-name"

    def test_estimate_size_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "m.gguf"
        f.write_bytes(b"x" * 1024)
        assert _estimate_model_size(str(f)) == pytest.approx(0.0, abs=1e-6)

    def test_estimate_size_from_filename(self) -> None:
        assert _estimate_model_size("/x/qwen2.5-7b.gguf") == 7.0
        assert _estimate_model_size("/x/llama3-8b.Q4.gguf") == 8.0

    def test_estimate_size_unknown(self) -> None:
        assert _estimate_model_size("/x/nothing-here.gguf") is None


# ---------------------------------------------------------------- recommend / profile

class TestRecommendAndProfile:
    _HW = SimpleNamespace(total_ram_gb=15.6, cpu_threads=8)

    def test_recommend_with_size(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        recs = {
            "gpu_layers": 0,
            "quant": "q4_k_m",
            "n_ctx": 2048,
            "threads": 4,
            "gpu_note": "cpu only",
            "quant_note": "balanced",
            "n_ctx_note": "fits ram",
            "threads_note": "all cores",
        }
        with (
            patch("fairbench.hardware.detect_hardware", return_value=self._HW),
            patch("fairbench.hardware.generate_recommendation", return_value=recs),
        ):
            result = runner.invoke(cli, ["recommend", "--model-size", "4.0"])
        assert result.exit_code == 0, result.output
        assert "n_gpu_layers: 0" in result.output

    def test_recommend_with_warning(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        recs = {
            "gpu_layers": 0,
            "quant": "q4_k_m",
            "n_ctx": 2048,
            "threads": 4,
            "gpu_note": "x",
            "quant_note": "x",
            "n_ctx_note": "x",
            "threads_note": "x",
            "warning": "model larger than RAM",
        }
        with (
            patch("fairbench.hardware.detect_hardware", return_value=self._HW),
            patch("fairbench.hardware.generate_recommendation", return_value=recs),
        ):
            result = runner.invoke(cli, ["recommend", "--model-size", "70.0"])
        assert result.exit_code == 0, result.output
        assert "model larger than RAM" in result.output

    def test_recommend_no_size(self, runner: CliRunner, cfg: BenchmarkConfig) -> None:
        with patch("fairbench.hardware.detect_hardware", return_value=self._HW):
            result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0, result.output
        assert "No model size provided" in result.output
