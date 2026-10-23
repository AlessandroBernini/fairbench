"""Backend for FairBench benchmarks."""

import time
import subprocess
import json
from pathlib import Path
from typing import Optional

import psutil


def detect_hardware() -> dict:
    """Detect system hardware."""
    info = {
        "cpu": _get_cpu_info(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": _get_gpu_info(),
        "os": _get_os_info(),
    }
    return info


def _get_cpu_info() -> dict:
    """Get CPU information."""
    return {
        "model": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
    }


def _get_gpu_info() -> dict:
    """Get GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return {
                "type": "nvidia",
                "model": lines[0].split(",")[0].strip() if lines else "unknown",
                "vram_gb": round(float(lines[0].split(",")[1].strip()) / 1024, 1) if len(lines) > 0 else 0,
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        pass
    return {"type": "none"}


def _get_os_info() -> str:
    """Get OS information."""
    import platform
    return f"{platform.system()} {platform.release()}"


def benchmark_speed(
    model_path: Path,
    prompt: str,
    n_tokens: int = 512,
    n_gpu_layers: int = -1,
    n_threads: int = 0,
) -> dict:
    """Benchmark inference speed using llama.cpp."""
    # Use llama.cpp CLI for speed measurement
    cmd = [
        "llama-bench",
        "-m", str(model_path),
        "-n", str(n_tokens),
        "-p", str(len(prompt)),
        "-ngl", str(n_gpu_layers),
        "-t", str(n_threads),
        "--output-json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fallback: manual measurement
    return _measure_speed_manual(model_path, prompt, n_tokens, n_gpu_layers, n_threads)


def _measure_speed_manual(
    model_path: Path,
    prompt: str,
    n_tokens: int,
    n_gpu_layers: int,
    n_threads: int,
) -> dict:
    """Manual speed measurement as fallback."""
    # This would use llama.cpp Python bindings
    # For now, return placeholder
    return {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "throughput": 0.0,
        "measured": False,
    }


def benchmark_memory(
    model_path: Path,
    n_gpu_layers: int = -1,
) -> dict:
    """Benchmark memory usage."""
    import psutil
    process = psutil.Process()

    # Load model and measure memory
    # This would use llama.cpp bindings to load the model
    return {
        "vram_peak_gb": 0.0,
        "cpu_ram_peak_gb": 0.0,
        "model_file_size_gb": round(model_path.stat().st_size / (1024**3), 2),
    }
