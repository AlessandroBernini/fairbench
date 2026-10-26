"""Configuration management for FairBench."""

from __future__ import annotations

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
    "results_dir": "/home/jarvispi/.fairbench/results",
}  # type: ignore[dict-item]

CONFIG_PATH = Path.home() / ".fairbench" / "config.yaml"


def load_config(path: str | None = None) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        return _default_config()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    return BenchmarkConfig(
        api_url=data.get("api_url", DEFAULT_CONFIG["api_url"]),
        default_n_ctx=data.get("default_n_ctx", DEFAULT_CONFIG["default_n_ctx"]),
        default_n_gpu_layers=data.get(
            "default_n_gpu_layers", DEFAULT_CONFIG["default_n_gpu_layers"]
        ),
        default_quant=data.get("default_quant", DEFAULT_CONFIG["default_quant"]),
        default_threads=data.get("default_threads", DEFAULT_CONFIG["default_threads"]),
        publish=data.get("publish", DEFAULT_CONFIG["publish"]),
        results_dir=data.get("results_dir", DEFAULT_CONFIG["results_dir"]),
        lite_mode=_parse_lite_mode(data.get("lite_mode")),
        swebench_instances=data.get("swebench_instances", 0),
        swebench_split=data.get("swebench_split", "full"),
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
