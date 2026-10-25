"""Shared fixtures for FairBench tests."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Temporary directory for benchmark results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def mock_config(tmp_results_dir: Path):
    """Mock FairBench Config to use temp directories."""
    config_data = {
        "api_url": "http://localhost:3000/api",
        "default_quant": "q4_k_m",
        "default_n_ctx": 4096,
        "default_n_gpu_layers": -1,
        "default_threads": 0,
        "publish": True,
        "results_dir": str(tmp_results_dir),
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    with patch("fairbench.config.Config._ensure_config"):
        from fairbench.config import Config
        config = Config.__new__(Config)
        config.config_path = config_path
        config._config = config_data
        yield config


@pytest.fixture
def mock_hardware():
    """Mock hardware detection results."""
    return {
        "cpu": {
            "model": "Test CPU",
            "cores": 4,
            "threads": 8,
        },
        "ram_gb": 16.0,
        "gpu": {
            "type": "none",
        },
        "os": "Linux 6.18",
    }


@pytest.fixture
def mock_model_file(tmp_path: Path) -> Path:
    """Create a fake GGUF model file for testing."""
    model_file = tmp_path / "test-model.gguf"
    model_file.write_bytes(b"\x00" * 1024)  # 1KB dummy file
    return model_file


@pytest.fixture
def mock_reporter(tmp_results_dir: Path):
    """Create a Reporter with temp directory."""
    from fairbench.reporter import Reporter
    return Reporter(results_dir=tmp_results_dir)
