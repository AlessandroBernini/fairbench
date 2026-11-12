"""Hardware detection and profiling for FairBench."""

from __future__ import annotations

import glob
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    """Detected hardware profile."""

    cpu_model: str = "Unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    total_ram_gb: float = 0.0
    gpu_available: bool = False
    gpu_type: Optional[str] = None  # "nvidia", "apple-silicon", "amd", None
    gpu_vram_gb: float = 0.0
    os_platform: str = "linux"  # "linux", "darwin", "windows"


def _detect_cpu() -> tuple[str, int, int]:
    """Detect CPU model, cores, and threads."""
    cpu_model = "Unknown"
    cpu_cores = 0
    cpu_threads = 0

    # Try psutil first (already a dependency)
    try:
        import psutil

        cpu_threads = psutil.cpu_count(logical=True) or 0
        cpu_cores = psutil.cpu_count(logical=False) or 0

        if cpu_cores == 0:
            cpu_cores = cpu_threads

        # Try to get CPU model name
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":")[1].strip()
                        break
            if not cpu_model or cpu_model == "Unknown":
                # Try lscpu
                try:
                    result = subprocess.run(
                        ["lscpu"], capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.splitlines():
                        if "Model name:" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cpu_model = result.stdout.strip() or "Apple CPU"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                cpu_model = "Apple CPU"
        elif platform.system() == "Windows":
            cpu_model = platform.processor() or "Intel/AMD CPU"

        return cpu_model, cpu_cores, cpu_threads

    except ImportError:
        pass

    # Fallback to platform module
    cpu_model = platform.processor() or "Unknown"
    try:
        cpu_threads = os.cpu_count() or 0
        cpu_cores = cpu_threads  # Assume same for fallback
    except Exception:
        pass

    return cpu_model, cpu_cores, cpu_threads


def _detect_ram() -> float:
    """Detect total RAM in GB."""
    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
        return round(total_bytes / (1024**3), 2)
    except ImportError:
        pass

    # Fallback
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
        except (FileNotFoundError, ValueError):
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            bytes_val = int(result.stdout.strip())
            return round(bytes_val / (1024**3), 2)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass

    return 0.0


def _detect_gpu_nvidia() -> tuple[bool, str | None, float]:
    """Detect NVIDIA GPU via nvidia-smi."""
    gpu_available = False
    gpu_type = None
    gpu_vram_gb = 0.0

    # Try common nvidia-smi paths
    nvidia_smi_paths = [
        "/usr/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "nvidia-smi",  # Check PATH
    ]

    for smi_path in nvidia_smi_paths:
        try:
            result = subprocess.run(
                [smi_path, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                # Parse first GPU
                parts = [line.strip() for line in lines[0].split(",")]
                if len(parts) >= 2:
                    try:
                        gpu_vram_gb = round(float(parts[1]) / 1024, 2)
                        return True, "nvidia", gpu_vram_gb
                    except ValueError:
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, PermissionError):
            continue

    return gpu_available, gpu_type, gpu_vram_gb


def _detect_gpu_apple() -> tuple[bool, str, float]:
    """Detect Apple Silicon GPU via system_profiler."""
    if platform.system() != "Darwin":
        return False, None, 0.0

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout
            # Check for Apple GPU indicators
            if "Apple" in output and ("GPU" in output or "Metal" in output):
                # Try to extract VRAM
                for line in output.splitlines():
                    if "VRAM" in line or "Display RAM" in line:
                        # e.g., "VRAM (Total): 16 GB"
                        match = re.search(r"([\d.]+)\s*(GB|MB)", line)
                        if match:
                            vram_val = float(match.group(1))
                            unit = match.group(2)
                            if unit == "MB":
                                vram_val /= 1024
                            return True, "apple-silicon", vram_val
                return True, "apple-silicon", 0.0  # Can't parse VRAM but GPU exists
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        pass

    return False, None, 0.0


def _detect_gpu_amd() -> tuple[bool, str, float]:
    """Detect AMD GPU via /sys/class/drm."""
    if platform.system() != "Linux":
        return False, None, 0.0

    try:
        vendor_files = glob.glob("/sys/class/drm/card*/device/vendor")
        for vf in vendor_files:
            with open(vf, "r") as f:
                vendor = f.read().strip()
                if vendor == "0x1002":  # AMD vendor ID
                    # Try to get VRAM from card*/device/memory_size or similar
                    card_dir = os.path.dirname(vf)
                    for size_file in [
                        "memory_size",
                        "resource0",
                    ]:
                        size_path = os.path.join(card_dir, size_file)
                        if os.path.exists(size_path):
                            with open(size_path, "r") as sf:
                                size_bytes = int(sf.read().strip(), 16)
                                return True, "amd", round(size_bytes / (1024**3), 2)
                    return True, "amd", 0.0  # Found GPU but can't read VRAM
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return False, None, 0.0


def detect_hardware() -> HardwareProfile:
    """Detect system hardware and return a HardwareProfile.

    Returns a HardwareProfile with detected CPU, RAM, GPU, and OS info.
    All detection is graceful — missing tools or permissions result in
    safe defaults (0 values, None) rather than exceptions.
    """
    os_platform = platform.system().lower()
    if os_platform not in ("linux", "darwin", "windows"):
        os_platform = "linux"

    # Detect CPU and RAM (always available)
    cpu_model, cpu_cores, cpu_threads = _detect_cpu()
    total_ram_gb = _detect_ram()

    # Detect GPU (try each vendor)
    gpu_available, gpu_type, gpu_vram_gb = _detect_gpu_nvidia()
    if not gpu_available:
        gpu_available, gpu_type, gpu_vram_gb = _detect_gpu_apple()
    if not gpu_available:
        gpu_available, gpu_type, gpu_vram_gb = _detect_gpu_amd()

    return HardwareProfile(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        total_ram_gb=total_ram_gb,
        gpu_available=gpu_available,
        gpu_type=gpu_type,
        gpu_vram_gb=gpu_vram_gb,
        os_platform=os_platform,
    )


def generate_adaptive_config(hw: HardwareProfile) -> dict:
    """Generate adaptive config based on hardware profile.

    Returns a dict with keys matching config.yaml fields:
    - default_n_ctx: int
    - default_n_gpu_layers: int
    - default_threads: int
    - default_quant: str
    """
    config = {}

    # Context window — scale with RAM (512 MB per 1024 context units)
    # Range: 4096 to 8192
    ctx = int(hw.total_ram_gb * 512)
    config["default_n_ctx"] = min(8192, max(4096, ctx))

    # GPU layers
    if hw.gpu_available:
        config["default_n_gpu_layers"] = -1  # offload all to GPU
    else:
        config["default_n_gpu_layers"] = 0

    # Threads — use ~75% of available cores, minimum 4
    threads = max(4, int(hw.cpu_threads * 0.75))
    config["default_threads"] = threads

    # Quantization suggestion based on hardware tier
    if hw.total_ram_gb < 8:
        config["default_quant"] = "q4_k_m"  # tightest for low RAM
    elif hw.total_ram_gb < 16:
        config["default_quant"] = "q5_k_m"
    else:
        config["default_quant"] = "q6_k"  # higher quality for ample RAM

    return config


def generate_recommendation(
    hw: HardwareProfile, model_size_gb: float
) -> dict:
    """Generate recommendations for a specific model size.

    Args:
        hw: HardwareProfile from detect_hardware()
        model_size_gb: Estimated model size in GB

    Returns:
        Dict with recommendations and explanations
    """
    recommendations = {}

    # Check if model fits in RAM
    if hw.total_ram_gb > 0 and model_size_gb > 0:
        if model_size_gb > hw.total_ram_gb * 0.9:
            recommendations["warning"] = (
                f"Model ({model_size_gb:.1f} GB) is close to or exceeds available RAM "
                f"({hw.total_ram_gb:.1f} GB). Consider a smaller model or lower quantization."
            )
        else:
            recommendations["warning"] = None

    # GPU recommendations
    if hw.gpu_available:
        if hw.gpu_vram_gb > 0:
            if model_size_gb <= hw.gpu_vram_gb * 0.8:
                recommendations["gpu_layers"] = -1
                recommendations["gpu_note"] = (
                    f"Model fits in GPU VRAM ({hw.gpu_vram_gb:.1f} GB). "
                    "Full GPU offload recommended."
                )
            else:
                # Estimate layers needed
                recommendations["gpu_layers"] = "partial"
                recommendations["gpu_note"] = (
                    f"Model ({model_size_gb:.1f} GB) exceeds GPU VRAM ({hw.gpu_vram_gb:.1f} GB). "
                    "Partial offload needed — use --n-gpu-layers to tune."
                )
        else:
            recommendations["gpu_layers"] = -1
            recommendations["gpu_note"] = "GPU detected, VRAM unknown. Try full offload first."
    else:
        recommendations["gpu_layers"] = 0
        recommendations["gpu_note"] = "No GPU detected. Running on CPU."

    # Quantization recommendation
    if hw.total_ram_gb < 8:
        recommendations["quant"] = "q4_k_m"
        recommendations["quant_note"] = "Low RAM — tightest quantization recommended."
    elif hw.total_ram_gb < 16:
        recommendations["quant"] = "q5_k_m"
        recommendations["quant_note"] = "Moderate RAM — balanced quantization."
    else:
        recommendations["quant"] = "q6_k"
        recommendations["quant_note"] = "Ample RAM — higher quality quantization."

    # Context window
    ctx = min(8192, max(4096, int(hw.total_ram_gb * 512)))
    recommendations["n_ctx"] = ctx
    recommendations["n_ctx_note"] = (
        f"Context window scaled to RAM: {ctx} (range: 4096-8192)"
    )

    # Threads
    threads = max(4, int(hw.cpu_threads * 0.75))
    recommendations["threads"] = threads
    recommendations["threads_note"] = (
        f"Threads: {threads} ({hw.cpu_threads} cores × 0.75)"
    )

    return recommendations
