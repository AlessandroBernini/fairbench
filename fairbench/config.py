"""Configuration management for FairBench."""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from fairbench.types import BenchmarkConfig, LiteMode

DEFAULT_CONFIG: dict[str, str | int | float | bool] = {
    "api_url": "https://fairbench.vercel.app/api",
    "default_n_ctx": 4096,
    "default_n_gpu_layers": -1,
    "default_quant": "q4_k_m",
    "default_threads": 0,
    "publish": True,
    "results_dir": str(Path.home() / ".fairbench" / "results"),
}  # type: ignore[dict-item]

CONFIG_PATH = Path.home() / ".fairbench" / "config.yaml"

# Config schema: key -> {type, default, validate}
VALID_CONFIG_KEYS: frozenset[str] = frozenset({
    "api_url",
    "default_n_ctx",
    "default_n_gpu_layers",
    "default_quant",
    "default_threads",
    "publish",
    "results_dir",
    "lite_mode",
    "swebench_instances",
    "swebench_split",
})

VALID_QUANTIZATIONS: frozenset[str] = frozenset({
    "q4_k_m", "q3_k_m", "q5_k_m", "q6_k", "q8_0", "f16", "f32",
})

VALID_SWEBENCH_SPLITS: frozenset[str] = frozenset({"full", "lite", "verified"})


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


def _is_valid_quant(value: str) -> bool:
    """Check if quantization string is valid."""
    return value in VALID_QUANTIZATIONS


def _is_valid_split(value: str) -> bool:
    """Check if swebench split is valid."""
    return value in VALID_SWEBENCH_SPLITS


def _coerce_bool(value: object) -> bool:
    """Coerce various truthy/falsy values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def validate_config(data: dict[str, object]) -> dict[str, object]:
    """Validate and coerce config values. Raises ConfigValidationError on bad input."""
    validated: dict[str, object] = {}

    # api_url
    api_url = data.get("api_url", DEFAULT_CONFIG["api_url"])
    if not isinstance(api_url, str) or not api_url:
        raise ConfigValidationError("api_url: must be a non-empty string")
    validated["api_url"] = api_url

    # default_n_ctx
    n_ctx = data.get("default_n_ctx", DEFAULT_CONFIG["default_n_ctx"])
    if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
        raise ConfigValidationError(
            f"default_n_ctx: must be a positive integer, got {n_ctx!r}"
        )
    validated["default_n_ctx"] = n_ctx

    # default_n_gpu_layers
    n_gpu = data.get("default_n_gpu_layers", DEFAULT_CONFIG["default_n_gpu_layers"])
    if not isinstance(n_gpu, int) or isinstance(n_gpu, bool):
        raise ConfigValidationError(
            f"default_n_gpu_layers: must be an integer, got {n_gpu!r}"
        )
    if n_gpu < -1:
        raise ConfigValidationError(
            f"default_n_gpu_layers: must be >= -1, got {n_gpu}"
        )
    validated["default_n_gpu_layers"] = n_gpu

    # default_quant
    quant = data.get("default_quant", DEFAULT_CONFIG["default_quant"])
    if not isinstance(quant, str) or not _is_valid_quant(quant):
        raise ConfigValidationError(
            f"default_quant: must be one of {sorted(VALID_QUANTIZATIONS)}, got {quant!r}"
        )
    validated["default_quant"] = quant

    # default_threads
    threads = data.get("default_threads", DEFAULT_CONFIG["default_threads"])
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise ConfigValidationError(
            f"default_threads: must be a non-negative integer, got {threads!r}"
        )
    validated["default_threads"] = threads

    # publish
    publish = data.get("publish", DEFAULT_CONFIG["publish"])
    validated["publish"] = _coerce_bool(publish)

    # results_dir
    results_dir = data.get("results_dir", DEFAULT_CONFIG["results_dir"])
    if not isinstance(results_dir, str) or not results_dir:
        raise ConfigValidationError("results_dir: must be a non-empty string")
    validated["results_dir"] = results_dir

    # lite_mode
    lite_mode_raw: str | None = data.get("lite_mode")  # type: ignore[assignment]
    validated["lite_mode"] = _parse_lite_mode(lite_mode_raw)

    # swebench_instances
    swebench_inst = data.get("swebench_instances", 0)
    if not isinstance(swebench_inst, int) or isinstance(swebench_inst, bool) or swebench_inst < 0:
        raise ConfigValidationError(
            f"swebench_instances: must be a non-negative integer, got {swebench_inst!r}"
        )
    validated["swebench_instances"] = swebench_inst

    # swebench_split
    swebench_split = data.get("swebench_split", "full")
    if not isinstance(swebench_split, str) or not _is_valid_split(swebench_split):
        raise ConfigValidationError(
            f"swebench_split: must be one of {sorted(VALID_SWEBENCH_SPLITS)}, got {swebench_split!r}"
        )
    validated["swebench_split"] = swebench_split

    # Warn on unknown keys
    unknown = set(data.keys()) - VALID_CONFIG_KEYS
    if unknown:
        warnings.warn(
            f"Unknown config keys ignored: {sorted(unknown)}",
            stacklevel=2,
        )

    return validated


def load_config(path: str | None = None) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        return _default_config()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    validated = validate_config(data)
    # validated values are guaranteed correct by validate_config
    return BenchmarkConfig(
        api_url=str(validated["api_url"]),
        default_n_ctx=int(validated["default_n_ctx"]),  # type: ignore[arg-type, call-overload]
        default_n_gpu_layers=int(validated["default_n_gpu_layers"]),  # type: ignore[arg-type, call-overload]
        default_quant=str(validated["default_quant"]),
        default_threads=int(validated["default_threads"]),  # type: ignore[arg-type, call-overload]
        publish=bool(validated["publish"]),
        results_dir=str(validated["results_dir"]),
        lite_mode=validated["lite_mode"],  # type: ignore[arg-type, assignment]
        swebench_instances=int(validated["swebench_instances"]),  # type: ignore[arg-type, call-overload]
        swebench_split=str(validated["swebench_split"]),
    )


def save_config(config: BenchmarkConfig, path: str | None = None) -> None:
    """Save benchmark configuration to YAML file."""
    config_path = Path(path) if path else CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "api_url": config.api_url,
        "default_n_ctx": config.default_n_ctx,
        "default_n_gpu_layers": config.default_n_gpu_layers,
        "default_quant": config.default_quant,
        "default_threads": config.default_threads,
        "publish": config.publish,
        "results_dir": config.results_dir,
    }

    if config.lite_mode != LiteMode.AUTO:
        data["lite_mode"] = config.lite_mode.value

    if config.swebench_instances > 0:
        data["swebench_instances"] = config.swebench_instances

    if config.swebench_split != "full":
        data["swebench_split"] = config.swebench_split

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _default_config() -> BenchmarkConfig:
    """Return default configuration."""
    return BenchmarkConfig(
        api_url=str(DEFAULT_CONFIG["api_url"]),
        default_n_ctx=int(DEFAULT_CONFIG["default_n_ctx"]),
        default_n_gpu_layers=int(DEFAULT_CONFIG["default_n_gpu_layers"]),
        default_quant=str(DEFAULT_CONFIG["default_quant"]),
        default_threads=int(DEFAULT_CONFIG["default_threads"]),
        publish=bool(DEFAULT_CONFIG["publish"]),
        results_dir=str(DEFAULT_CONFIG["results_dir"]),
    )


def _parse_lite_mode(value: str | None) -> LiteMode:
    """Parse lite mode from string value."""
    if value is None:
        return LiteMode.AUTO
    mapping = {
        "disabled": LiteMode.DISABLED,
        "enabled": LiteMode.ENABLED,
        "auto": LiteMode.AUTO,
    }
    return mapping.get(value.lower(), LiteMode.AUTO)


def ensure_results_dir(results_dir: str | None = None) -> Path:
    """Ensure results directory exists and return it."""
    dir_path = Path(results_dir) if results_dir else CONFIG_PATH.parent / "results"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
