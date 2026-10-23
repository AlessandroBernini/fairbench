"""Configuration for FairBench."""

from pathlib import Path
from typing import Optional

import yaml


class Config:
    """FairBench configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".fairbench" / "config.yaml"
        self._ensure_config()

    def _ensure_config(self):
        """Create default config if it doesn't exist."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            default = {
                "api_url": "https://fairbench.vercel.app/api",
                "default_quant": "q4_k_m",
                "default_n_ctx": 4096,
                "default_n_gpu_layers": -1,
                "default_threads": 0,  # 0 = auto
                "publish": True,
                "results_dir": str(Path.home() / ".fairbench" / "results"),
            }
            self.config_path.write_text(yaml.dump(default, default_flow_style=False))

    @property
    def api_url(self) -> str:
        return self._get("api_url")

    @property
    def default_quant(self) -> str:
        return self._get("default_quant")

    @property
    def default_n_ctx(self) -> int:
        return self._get("default_n_ctx")

    @property
    def default_n_gpu_layers(self) -> int:
        return self._get("default_n_gpu_layers")

    @property
    def default_threads(self) -> int:
        return self._get("default_threads")

    @property
    def publish_default(self) -> bool:
        return self._get("publish")

    @property
    def results_dir(self) -> Path:
        path = Path(self._get("results_dir"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get(self, key: str):
        """Get a config value."""
        config = yaml.safe_load(self.config_path.read_text())
        return config.get(key)
