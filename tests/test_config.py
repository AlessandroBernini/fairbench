"""Tests for fairbench.config — configuration validation, load/save."""

from pathlib import Path

import pytest

from fairbench.config import (
    ConfigValidationError,
    _coerce_bool,
    _is_valid_quant,
    _is_valid_split,
    _parse_lite_mode,
    ensure_results_dir,
    load_config,
    save_config,
    validate_config,
)
from fairbench.types import BenchmarkConfig, LiteMode


class TestCoerceBool:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            (1, True),
            (0, False),
        ],
    )
    def test_coerce_bool_various_inputs(self, value, expected):
        assert _coerce_bool(value) is expected


class TestIsValidQuant:
    @pytest.mark.parametrize("quant,expected", [("q4_k_m", True), ("q5_k_m", True), ("q8_0", True), ("invalid", False)])
    def test_is_valid_quant(self, quant, expected):
        assert _is_valid_quant(quant) is expected


class TestIsValidSplit:
    @pytest.mark.parametrize("split,expected", [("full", True), ("lite", True), ("verified", True), ("test", False)])
    def test_is_valid_split(self, split, expected):
        assert _is_valid_split(split) is expected


class TestParseLiteMode:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, LiteMode.AUTO),
            ("disabled", LiteMode.DISABLED),
            ("enabled", LiteMode.ENABLED),
            ("auto", LiteMode.AUTO),
            ("invalid", LiteMode.AUTO),
        ],
    )
    def test_parse_lite_mode(self, raw, expected):
        assert _parse_lite_mode(raw) is expected


class TestValidateConfig:
    def test_validate_defaults(self):
        data = {}
        validated = validate_config(data)
        assert validated["default_n_ctx"] == 4096
        assert validated["default_n_gpu_layers"] == -1
        assert validated["default_quant"] == "q4_k_m"
        assert validated["publish"] is True

    def test_validate_valid_custom(self):
        data = {
            "api_url": "https://example.com",
            "default_n_ctx": 2048,
            "default_n_gpu_layers": 4,
            "default_quant": "q5_k_m",
            "default_threads": 2,
            "publish": "yes",
            "results_dir": "/tmp/results",
            "lite_mode": "auto",
            "swebench_instances": 100,
            "swebench_split": "full",
        }
        validated = validate_config(data)
        assert validated["default_n_ctx"] == 2048
        assert validated["default_n_gpu_layers"] == 4
        assert validated["default_quant"] == "q5_k_m"
        assert validated["publish"] is True
        assert validated["lite_mode"] is LiteMode.AUTO

    def test_validate_invalid_n_ctx(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_config({"default_n_ctx": -1})
        assert "default_n_ctx" in str(exc.value)

    def test_validate_invalid_n_gpu_layers(self):
        with pytest.raises(ConfigValidationError):
            validate_config({"default_n_gpu_layers": -2})

    def test_validate_invalid_quant(self):
        with pytest.raises(ConfigValidationError):
            validate_config({"default_quant": "q99"})

    def test_validate_invalid_threads(self):
        with pytest.raises(ConfigValidationError):
            validate_config({"default_threads": -1})

    def test_validate_invalid_split(self):
        with pytest.raises(ConfigValidationError):
            validate_config({"swebench_split": "unknown"})


class TestLoadSaveConfig:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        cfg = BenchmarkConfig(
            api_url="https://api.test",
            default_n_ctx=1024,
            default_n_gpu_layers=2,
            default_quant="q5_k_m",
            default_threads=1,
            publish=False,
            results_dir=str(tmp_path / "results"),
            lite_mode=LiteMode.ENABLED,
            swebench_instances=50,
            swebench_split="full",
        )
        config_file = tmp_path / "config.yaml"
        save_config(cfg, str(config_file))
        loaded = load_config(str(config_file))
        assert loaded.api_url == cfg.api_url
        assert loaded.default_n_ctx == cfg.default_n_ctx
        assert loaded.publish is False
        assert loaded.lite_mode is LiteMode.ENABLED

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        # load_config should fall back to defaults when file missing
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.default_n_ctx == 4096


class TestEnsureResultsDir:
    def test_ensure_creates_directory(self, tmp_path: Path):
        new_dir = tmp_path / "new_results"
        result = ensure_results_dir(str(new_dir))
        assert result.exists()
        assert result == new_dir
