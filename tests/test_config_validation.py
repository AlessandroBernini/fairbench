"""Tests for fairbench.config — config schema validation."""

from __future__ import annotations

import warnings

import pytest

from fairbench.config import (
    VALID_QUANTIZATIONS,
    VALID_SWEBENCH_SPLITS,
    ConfigValidationError,
    _coerce_bool,
    _is_valid_quant,
    _is_valid_split,
    validate_config,
)


class TestCoerceBool:
    def test_true_strings(self):
        """_coerce_bool handles true-like strings."""
        for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert _coerce_bool(val) is True

    def test_false_strings(self):
        """_coerce_bool handles false-like strings."""
        for val in ("false", "False", "FALSE", "0", "no", "No"):
            assert _coerce_bool(val) is False

    def test_bool_passthrough(self):
        """_coerce_bool passes through bool values."""
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False

    def test_int_coercion(self):
        """_coerce_bool coerces integers."""
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False

    def test_float_coercion(self):
        """_coerce_bool coerces floats."""
        assert _coerce_bool(1.0) is True
        assert _coerce_bool(0.0) is False

    def test_none_returns_false(self):
        """_coerce_bool returns False for None."""
        assert _coerce_bool(None) is False


class TestValidationHelpers:
    def test_is_valid_quant_true(self):
        """_is_valid_quant returns True for valid quantizations."""
        for quant in VALID_QUANTIZATIONS:
            assert _is_valid_quant(quant) is True

    def test_is_valid_quant_false(self):
        """_is_valid_quant returns False for invalid quantizations."""
        assert _is_valid_quant("invalid") is False
        assert _is_valid_quant("") is False

    def test_is_valid_split_true(self):
        """_is_valid_split returns True for valid splits."""
        for split in VALID_SWEBENCH_SPLITS:
            assert _is_valid_split(split) is True

    def test_is_valid_split_false(self):
        """_is_valid_split returns False for invalid splits."""
        assert _is_valid_split("invalid") is False
        assert _is_valid_split("") is False


class TestValidateConfig:
    def test_valid_config(self):
        """validate_config accepts valid config."""
        data = {
            "api_url": "https://example.com/api",
            "default_n_ctx": 8192,
            "default_n_gpu_layers": -1,
            "default_quant": "q6_k",
            "default_threads": 8,
            "publish": True,
            "results_dir": "/tmp/results",
            "lite_mode": "enabled",
            "swebench_instances": 10,
            "swebench_split": "lite",
        }
        result = validate_config(data)
        assert result["api_url"] == "https://example.com/api"
        assert result["default_n_ctx"] == 8192
        assert result["publish"] is True

    def test_default_values(self):
        """validate_config applies defaults for missing keys."""
        data = {}
        result = validate_config(data)
        assert result["default_n_ctx"] == 4096
        assert result["default_n_gpu_layers"] == -1
        assert result["default_quant"] == "q4_k_m"
        assert result["default_threads"] == 0
        assert result["publish"] is True
        assert result["swebench_split"] == "full"

    def test_publish_string_true(self):
        """validate_config coerces publish=true to bool True."""
        data = {"publish": "true"}
        result = validate_config(data)
        assert result["publish"] is True

    def test_publish_string_false(self):
        """validate_config coerces publish=false to bool False."""
        data = {"publish": "false"}
        result = validate_config(data)
        assert result["publish"] is False

    def test_invalid_n_ctx(self):
        """validate_config rejects invalid n_ctx."""
        data = {"default_n_ctx": -1}
        with pytest.raises(ConfigValidationError, match="default_n_ctx"):
            validate_config(data)

    def test_invalid_gpu_layers(self):
        """validate_config rejects invalid n_gpu_layers."""
        data = {"default_n_gpu_layers": -2}
        with pytest.raises(ConfigValidationError, match="default_n_gpu_layers"):
            validate_config(data)

    def test_invalid_quant(self):
        """validate_config rejects invalid quantization."""
        data = {"default_quant": "invalid"}
        with pytest.raises(ConfigValidationError, match="default_quant"):
            validate_config(data)

    def test_invalid_threads(self):
        """validate_config rejects invalid threads."""
        data = {"default_threads": -1}
        with pytest.raises(ConfigValidationError, match="default_threads"):
            validate_config(data)

    def test_invalid_split(self):
        """validate_config rejects invalid swebench_split."""
        data = {"swebench_split": "invalid"}
        with pytest.raises(ConfigValidationError, match="swebench_split"):
            validate_config(data)

    def test_warns_on_unknown_keys(self):
        """validate_config warns on unknown keys."""
        data = {"unknown_key": "value", "api_url": "https://example.com"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_config(data)
            assert len(w) == 1
            assert "unknown_key" in str(w[0].message)

    def test_api_url_empty_string(self):
        """validate_config rejects empty api_url."""
        data = {"api_url": ""}
        with pytest.raises(ConfigValidationError, match="api_url"):
            validate_config(data)

    def test_results_dir_empty_string(self):
        """validate_config rejects empty results_dir."""
        data = {"results_dir": ""}
        with pytest.raises(ConfigValidationError, match="results_dir"):
            validate_config(data)

    def test_publish_boolean_coercion(self):
        """validate_config coerces various publish values."""
        for val, expected in (("true", True), ("false", False), (True, True), (False, False), (1, True), (0, False)):
            data = {"publish": val}
            result = validate_config(data)
            assert result["publish"] is expected, f"publish={val!r} -> {result['publish']}"
