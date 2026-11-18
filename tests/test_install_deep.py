"""Deeper coverage tests for fairbench.install (F6)."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from fairbench.config import BenchmarkConfig
from fairbench.install import (
    check_dependencies,
    cmd_download,
    cmd_init,
    cmd_quick_start,
    cmd_verify,
)


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    with patch("pathlib.Path.home", return_value=home):
        yield home


# ---------------------------------------------------------------- check_dependencies

class TestCheckDependencies:
    def test_all_present(self) -> None:
        mods = {m: MagicMock() for m in ("lm_eval", "llama_cpp", "swebench")}
        with patch.dict("sys.modules", mods):
            deps = check_dependencies()
        assert all("✓" in v for v in deps.values())

    def test_all_missing(self) -> None:
        with patch.dict("sys.modules", {m: None for m in ("lm_eval", "llama_cpp", "swebench")}):
            deps = check_dependencies()
        assert all("✗" in v for v in deps.values())


# ---------------------------------------------------------------- cmd_verify

class TestCmdVerify:
    def test_missing_config_dir_warns(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        with patch("fairbench.install.check_dependencies", return_value={"lm-eval": "✓ ok"}):
            cmd_verify()
        out = capsys.readouterr().out
        assert "Config directory missing" in out

    def test_full_ok_path(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        (fake_home / ".fairbench").mkdir()
        (fake_home / ".fairbench" / "config.yaml").write_text("publish: true\n")
        deps = {k: "✓ Installed" for k in ("lm-eval", "llama-cpp-python", "swebench")}
        with patch("fairbench.install.check_dependencies", return_value=deps):
            cmd_verify()
        out = capsys.readouterr().out
        assert "ready to run FairBench" in out

    def test_low_disk_space_warning(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        deps = {k: "✓ Installed" for k in ("lm-eval", "llama-cpp-python", "swebench")}
        with (
            patch("fairbench.install.check_dependencies", return_value=deps),
            patch("shutil.disk_usage", return_value=(100, 96, 2 * 1024**3)),
        ):
            cmd_verify()
        out = capsys.readouterr().out
        assert "Low disk space" in out

    def test_disk_usage_oserror_swallowed(self, fake_home: Path) -> None:
        with (
            patch("fairbench.install.check_dependencies", return_value={"lm-eval": "✓ ok"}),
            patch("shutil.disk_usage", side_effect=OSError("no stat")),
        ):
            cmd_verify()  # must not raise


# ---------------------------------------------------------------- cmd_quick_start

class TestCmdQuickStart:
    def test_model_present_with_llama(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        models = fake_home / "models"
        models.mkdir()
        (models / "qwen2.5-1.5b-instruct-q4_k_m.gguf").write_bytes(b"gguf")
        deps = {"lm-eval": "✓ Installed", "swebench": "✗ Not installed"}
        with (
            patch.dict("sys.modules", {"llama_cpp": MagicMock()}),
            patch("fairbench.install.check_dependencies", return_value=deps),
        ):
            cmd_quick_start()
        out = capsys.readouterr().out
        assert "Speed benchmark" in out
        assert "Quality benchmarks (lite mode)" in out

    def test_model_present_without_llama(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        models = fake_home / "models"
        models.mkdir()
        (models / "qwen2.5-1.5b-instruct-q4_k_m.gguf").write_bytes(b"gguf")
        deps = {"lm-eval": "✗ Not installed", "swebench": "✓ Installed"}
        with (
            patch.dict("sys.modules", {"llama_cpp": None}),
            patch("fairbench.install.check_dependencies", return_value=deps),
        ):
            cmd_quick_start()
        out = capsys.readouterr().out
        assert "llama-cpp-python not installed" in out
        assert "SWE-bench" in out


# ---------------------------------------------------------------- cmd_init

def _hw_mock() -> MagicMock:
    hw = MagicMock()
    hw.cpu_model = "TestCPU"
    hw.cpu_cores = 4
    hw.cpu_threads = 8
    hw.total_ram_gb = 16.0
    hw.gpu_available = True
    hw.gpu_type = "nvidia"
    hw.gpu_vram_gb = 8.0
    return hw


def _adaptive() -> dict:
    return {
        "default_n_ctx": 4096,
        "default_n_gpu_layers": 35,
        "default_threads": 4,
        "default_quant": "q5_k_m",
    }


class TestCmdInit:
    def test_fresh_config_hardware_ok(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        cfg = BenchmarkConfig(results_dir=str(fake_home / "results"))
        with (
            patch("click.prompt", side_effect=["myuser", 35, "q5_k_m"]),
            patch("click.confirm", return_value=True),
            patch("fairbench.config.load_config", return_value=cfg),
            patch("fairbench.config.save_config") as save,
            patch("fairbench.hardware.detect_hardware", return_value=_hw_mock()),
            patch("fairbench.hardware.generate_adaptive_config", return_value=_adaptive()),
        ):
            cmd_init()
        out = capsys.readouterr().out
        assert "Creating new configuration" in out
        assert "Recommended settings" in out
        save.assert_called_once()
        assert cfg.api_url.endswith("/api/myuser")

    def test_existing_config_hardware_fail(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        config_dir = fake_home / ".fairbench"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("publish: true\n")
        existing = BenchmarkConfig(api_url="https://fairbench.vercel.app/api/olduser")
        with (
            patch("click.prompt", side_effect=["", -1, "q4_k_m"]) as prompt,
            patch("click.confirm", return_value=False),
            patch("fairbench.config.load_config", return_value=existing),
            patch("fairbench.config.save_config"),
            patch("fairbench.hardware.detect_hardware", side_effect=OSError("no lspci")),
        ):
            cmd_init()
        out = capsys.readouterr().out
        assert "Could not analyze hardware" in out
        assert "Updating existing configuration" in out
        # username default extracted from existing api_url
        assert prompt.call_args_list[0].kwargs.get("default") == "olduser"

    def test_adaptive_defaults_rejected(self, fake_home: Path, capsys: pytest.CaptureFixture) -> None:
        cfg = BenchmarkConfig(results_dir=str(fake_home / "results"))
        with (
            patch("click.prompt", side_effect=["u", 35, "q5_k_m"]),
            patch("click.confirm", return_value=False),
            patch("fairbench.config.load_config", return_value=cfg),
            patch("fairbench.config.save_config"),
            patch("fairbench.hardware.detect_hardware", return_value=_hw_mock()),
            patch("fairbench.hardware.generate_adaptive_config", return_value=_adaptive()),
        ):
            cmd_init()
        out = capsys.readouterr().out
        assert "Using your customized values" in out


# ---------------------------------------------------------------- cmd_download

def _fake_hf(list_files_ret: list[str], download_ret: str = "/tmp/model.gguf") -> dict[str, ModuleType]:
    """Fake huggingface_hub + tqdm so the primary path is exercised deterministically."""
    hf = ModuleType("huggingface_hub")
    hf.list_repo_files = MagicMock(return_value=list_files_ret)  # type: ignore[attr-defined]
    hf.hf_hub_download = MagicMock(return_value=download_ret)  # type: ignore[attr-defined]
    tq = ModuleType("tqdm")
    tq.tqdm = MagicMock()  # type: ignore[attr-defined]
    return {"huggingface_hub": hf, "tqdm": tq}


class TestCmdDownload:
    def test_auto_select_first_gguf(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        fakes = _fake_hf(["README.md", "model-Q4_K_M.gguf", "model-Q8_0.gguf"])
        with patch.dict("sys.modules", fakes):
            cmd_download("Qwen/Test", local_dir=str(tmp_path / "dl"))
        assert "Auto-selecting: model-Q4_K_M.gguf" in capsys.readouterr().out
        fakes["huggingface_hub"].hf_hub_download.assert_called_once()

    def test_no_gguf_files(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        fakes = _fake_hf(["README.md", "config.json"])
        with patch.dict("sys.modules", fakes):
            cmd_download("Qwen/Test", local_dir=str(tmp_path / "dl"))
        assert "No GGUF files found" in capsys.readouterr().out

    def test_specific_file_not_found(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        fakes = _fake_hf(["a-Q4.gguf"])
        with patch.dict("sys.modules", fakes):
            cmd_download("Qwen/Test", filename="missing.gguf", local_dir=str(tmp_path / "dl"))
        out = capsys.readouterr().out
        assert "not found, available files" in out
        fakes["huggingface_hub"].hf_hub_download.assert_not_called()

    def test_specific_file_ok(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        fakes = _fake_hf(["a-Q4.gguf"], download_ret="/x/a-Q4.gguf")
        with patch.dict("sys.modules", fakes):
            cmd_download("Qwen/Test", filename="a-Q4.gguf", local_dir=str(tmp_path / "dl"))
        assert "downloaded successfully" in capsys.readouterr().out

    def test_cli_fallback_success(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0))
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            patch("subprocess.run", run),
        ):
            cmd_download("Qwen/Test", filename="m.gguf", local_dir=str(tmp_path / "dl"))
        out = capsys.readouterr().out
        assert "Using huggingface-cli" in out
        assert "downloaded successfully" in out

    def test_cli_fallback_no_filename(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            cmd_download("Qwen/Test", local_dir=str(tmp_path / "dl"))
        assert "Use --filename" in capsys.readouterr().out

    def test_cli_fallback_returncode_fail(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        run = MagicMock(return_value=MagicMock(returncode=1))
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            patch("subprocess.run", run),
        ):
            cmd_download("Qwen/Test", filename="m.gguf", local_dir=str(tmp_path / "dl"))
        assert "Download failed" in capsys.readouterr().out

    def test_cli_fallback_binary_missing(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            patch("subprocess.run", side_effect=FileNotFoundError()),
        ):
            cmd_download("Qwen/Test", filename="m.gguf", local_dir=str(tmp_path / "dl"))
        assert "huggingface-cli not found" in capsys.readouterr().out
