"""Docker container support for FairBench."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def detect_docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_gpu_support() -> str:
    """Detect GPU support in Docker environment.

    Returns:
        'nvidia' for NVIDIA GPU support,
        'rocm' for AMD ROCm support,
        'none' for no GPU support.
    """
    system = platform.system()

    # Check for nvidia-smi on Linux
    if system == "Linux":
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "nvidia"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for ROCm on Linux
        if os.path.exists("/opt/rocm/bin/rocminfo"):
            return "rocm"

    return "none"


def detect_macos_docker_limitation() -> bool:
    """Detect if running Docker on macOS (no GPU support)."""
    return platform.system() == "Darwin"


def build_docker_image(image_name: str = "fairbench") -> bool:
    """Build the FairBench Docker image.

    Returns True if build succeeded, False otherwise.
    """
    dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"

    if not dockerfile.exists():
        print(f"Dockerfile not found at {dockerfile}")
        return False

    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(dockerfile.parent)],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print(f"✅ Docker image built: {image_name}")
            return True
        else:
            print(f"❌ Docker build failed:\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out (10 min limit)")
        return False
    except (OSError, RuntimeError) as e:
        print(f"❌ Docker build error: {e}")
        return False


def run_in_docker(
    image_name: str,
    model_path: str,
    category: str = "all",
    lite_mode: str = "disabled",
    gpu_support: str = "none",
    ci_mode: bool = False,
) -> int:
    """Run FairBench benchmarks inside a Docker container.

    Args:
        image_name: Docker image to use (built or pulled).
        model_path: Path to GGUF model file (mounted from host).
        category: Benchmark category to run.
        lite_mode: Lite mode setting.
        gpu_support: Detected GPU support level.
        ci_mode: Run in CI mode (no TTY, no progress bars).

    Returns:
        Exit code from the container execution.
    """
    # Resolve model path for mount
    if model_path.startswith("~"):
        model_path = os.path.expanduser(model_path)

    model_dir = str(Path(model_path).parent)
    model_file = str(Path(model_path).name)

    # Prepare environment variables
    env = os.environ.copy()

    # Check GPU support
    gpu_args: list[str] = []
    if gpu_support == "nvidia":
        gpu_args.extend(["--gpus", "all"])
    elif gpu_support == "rocm":
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    # Build docker run command
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{model_dir}:/app/models:ro",
        "-v", f"{os.path.expanduser('~')}/.fairbench:/root/.fairbench",
    ] + gpu_args

    if ci_mode:
        cmd.append("--env")
        cmd.append("TERM=dumb")  # Disable progress bars in CI
    else:
        cmd.append("-it")  # Interactive mode with TTY

    cmd.extend([image_name, "fairbench", "run"])
    cmd.extend(["/app/models/", model_file])
    cmd.extend(["--category", category])
    cmd.extend(["--lite" if lite_mode == "enabled" else "--no-lite"])

    print("\n🐳 Running in Docker container...")
    print(f"   Image: {image_name}")
    print(f"   GPU: {'Enabled' if gpu_support != 'none' else 'CPU only'}")
    print(f"   Model: {model_file}")
    print("-" * 50)

    try:
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=False,
            timeout=3600,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("❌ Docker run timed out (1 hour limit)")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130


def verify_docker_setup() -> dict[str, str]:
    """Verify Docker installation and GPU support.

    Returns a dict with status information.
    """
    info: dict[str, str] = {}

    # Check Docker availability
    has_docker = detect_docker_available()
    info["docker_available"] = "Yes" if has_docker else "No"

    if has_docker:
        gpu = detect_gpu_support()
        info["gpu_support"] = gpu.capitalize() if gpu != "none" else "CPU only"

        if detect_macos_docker_limitation():
            info["note"] = "Docker on macOS has no GPU acceleration (CPU-only mode)"

    return info
