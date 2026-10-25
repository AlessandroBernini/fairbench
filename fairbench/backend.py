"""Backend for FairBench benchmarks."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
from rich.console import Console

console = Console()


def detect_hardware() -> dict[str, Any]:
    """Detect system hardware."""
    info = {
        "cpu": _get_cpu_info(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": _get_gpu_info(),
        "os": _get_os_info(),
    }
    return info


def _get_cpu_info() -> dict[str, Any]:
    """Get CPU information."""
    return {
        "model": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
    }


def _get_gpu_info() -> dict[str, Any]:
    """Get GPU information."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return {
                "type": "nvidia",
                "model": lines[0].split(",")[0].strip() if lines else "unknown",
                "vram_gb": round(float(lines[0].split(",")[1].strip()) / 1024 / 1024, 1)
                if len(lines) > 0
                else 0,
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
) -> dict[str, Any]:
    """Benchmark inference speed using llama-cpp-python.

    Measures TTFT (time to first token), TPOT (tokens per output token),
    and throughput (tokens/sec) using llama-cpp-python bindings for precise
    timing. Falls back to llama-bench CLI if available.

    Args:
        model_path: Path to the GGUF model file.
        prompt: Prompt string for evaluation.
        n_tokens: Number of tokens to generate.
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        n_threads: Number of CPU threads (0 = auto).

    Returns:
        dict with keys:
            - ttft_ms: Time to first token in milliseconds
            - tpot_ms: Tokens per output token in milliseconds
            - throughput: Throughput in tokens per second
            - measured: Whether a real measurement was performed
    """
    console.print(f"\n[bold blue]Speed benchmark:[/bold blue] {model_path.name}")
    console.print(f"  Tokens: {n_tokens}, Prompt len: {len(prompt)}, GPU layers: {n_gpu_layers}")

    # Try llama-bench CLI first (fastest if available)
    try:
        result = _benchmark_with_llama_bench(model_path, prompt, n_tokens, n_gpu_layers, n_threads)
        if result.get("measured"):
            console.print("  [bold green]llama-bench result:[/bold green]")
            console.print(f"    TTFT: {result['ttft_ms']:.2f} ms")
            console.print(f"    TPOT: {result['tpot_ms']:.2f} ms")
            console.print(f"    Throughput: {result['throughput']:.2f} tok/s")
            return result
    except Exception:
        pass  # Fall through to llama-cpp-python

    # Use llama-cpp-python bindings for precise timing
    return _benchmark_with_llama_cpp(model_path, prompt, n_tokens, n_gpu_layers, n_threads)


def _benchmark_with_llama_bench(
    model_path: Path,
    prompt: str,
    n_tokens: int,
    n_gpu_layers: int,
    n_threads: int,
) -> dict[str, Any]:
    """Try llama-bench CLI for speed measurement."""
    cmd = [
        "llama-bench",
        "-m", str(model_path),
        "-n", str(n_tokens),
        "-p", str(len(prompt)),
        "-ngl", str(n_gpu_layers),
        "-t", str(n_threads) if n_threads > 0 else "0",
        "--output-json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # llama-bench returns a list of benchmarks
            if isinstance(data, list) and len(data) > 0:
                bench = data[0]
                return {
                    "ttft_ms": float(bench.get("prompt_ms", 0) / max(bench.get("prompt_n", 1), 1)),
                    "tpot_ms": float(bench.get("prompt_ms", 0) / max(bench.get("prompt_n", 1), 1)),
                    "throughput": float(bench.get("speed", 0)),
                    "measured": True,
                }
            elif isinstance(data, dict):
                return {
                    "ttft_ms": float(data.get("prompt_ms", 0)),
                    "tpot_ms": float(data.get("n_keep_ms", 0)),
                    "throughput": float(data.get("speed", 0)),
                    "measured": True,
                }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        pass
    return {"ttft_ms": 0.0, "tpot_ms": 0.0, "throughput": 0.0, "measured": False}


def _benchmark_with_llama_cpp(
    model_path: Path,
    prompt: str,
    n_tokens: int,
    n_gpu_layers: int,
    n_threads: int,
) -> dict[str, Any]:
    """Benchmark using llama-cpp-python bindings with precise timing."""
    try:
        from llama_cpp import Llama
    except ImportError:
        console.print(
            "[yellow]Warning: llama-cpp-python not available "
            "for speed benchmark.[/yellow]"
        )
        return {"ttft_ms": 0.0, "tpot_ms": 0.0, "throughput": 0.0, "measured": False}

    try:
        console.print("[bold]Loading model for speed benchmark...[/bold]")
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads if n_threads > 0 else 0,
            n_ctx=4096,
        )

        # Measure TTFT (time to first token)
        console.print("[bold]Measuring TTFT...[/bold]")
        ttft_times = []
        for _ in range(3):  # Average over 3 runs
            start = time.perf_counter()
            llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            ttft_times.append(elapsed)
        ttft_ms = sum(ttft_times) / len(ttft_times)
        console.print(f"  TTFT: {ttft_ms:.2f} ms (avg of 3 runs)")

        # Measure TPOT and throughput
        console.print("[bold]Measuring throughput...[/bold]")
        throughput_times = []
        for _ in range(3):  # Average over 3 runs
            start = time.perf_counter()
            llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=n_tokens,
                temperature=0.7,
            )
            elapsed = (time.perf_counter() - start)
            throughput = n_tokens / elapsed if elapsed > 0 else 0
            throughput_times.append(throughput)

        avg_throughput = sum(throughput_times) / len(throughput_times)
        tpot_ms = (1.0 / avg_throughput * 1000) if avg_throughput > 0 else 0

        console.print(f"  Throughput: {avg_throughput:.2f} tok/s")
        console.print(f"  TPOT: {tpot_ms:.2f} ms")

        return {
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "throughput": round(avg_throughput, 2),
            "measured": True,
        }

    except Exception as e:
        console.print(f"[yellow]Warning: llama-cpp benchmark failed ({e}).[/yellow]")
        return {"ttft_ms": 0.0, "tpot_ms": 0.0, "throughput": 0.0, "measured": False}


def _detect_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available via nvidia-smi."""
    try:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _query_vram_usage() -> tuple[float, float]:
    """Query current VRAM usage in MB (used, total). Returns (used_mb, total_mb)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return (0, 0)
        line = result.stdout.strip().split(",")[0].strip()
        total_line = result.stdout.strip().split(",")[1].strip()
        used_mb = float(line)
        total_mb = float(total_line)
        return (used_mb, total_mb)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return (0, 0)


def _load_model(model_path: Path, n_gpu_layers: int = -1) -> None:
    """Load the model into memory using llama_cpp (if available).

    The Llama object is kept alive during function execution so peak
    memory usage is captured before the function returns.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        console.print(
            "[yellow]Warning: llama_cpp is not available. "
            "Model cannot be loaded; baseline measurements will be used.[/yellow]"
        )
        return

    Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_threads=0,
    )


def benchmark_memory(
    model_path: Path,
    n_gpu_layers: int = -1,
) -> dict[str, Any]:
    """Benchmark memory usage of a loaded model.

    Measures VRAM peak (via nvidia-smi) and CPU RAM peak (via psutil)
    during model loading. Supports CPU-only mode where VRAM = 0 when no
    NVIDIA GPU is available.

    Args:
        model_path: Path to the model file (e.g., GGUF file).
        n_gpu_layers: Number of layers to offload to GPU (-1 = all on CPU).

    Returns:
        dict with keys:
            - vram_peak_gb: float
            - cpu_ram_peak_gb: float
            - model_file_size_gb: float
    """
    # Validate model path
    if not model_path.is_file():
        console.print(f"[red]Error: model file not found or not a file: {model_path}[/red]")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_file_size_gb = round(model_path.stat().st_size / (1024 ** 3), 2)

    console.print(f"[bold]Memory benchmark starting:[/bold] {model_path}")
    console.print(f"  Model file size: {model_file_size_gb} GB")
    console.print(f"  GPU layers offloaded: {n_gpu_layers}")

    # --- Baseline measurements (before model load) ---
    gpu_available = _detect_gpu_available()
    before_used_mb, before_total_mb = (
        _query_vram_usage() if gpu_available else (0, 0)
    )
    before_rss = psutil.Process().memory_info().rss

    if gpu_available:
        console.print(
            f"  VRAM before load: {before_used_mb / 1024:.2f} / {before_total_mb / 1024:.2f} GB"
        )
    else:
        console.print("[yellow]No NVIDIA GPU detected (CPU-only mode). VRAM not measured.[/yellow]")

    console.print(f"  CPU RAM before load: {before_rss / 1024 / 1024:.2f} GB")

    # --- Load model ---
    try:
        _load_model(model_path, n_gpu_layers=n_gpu_layers)
    except Exception as e:
        console.print(
            f"[yellow]Warning: model loading failed ({e}). "
            "Using baseline measurements.[/yellow]"
        )

    # --- Peak measurements (after model load) ---
    after_used_mb, after_total_mb = (
        _query_vram_usage() if gpu_available else (0, 0)
    )
    after_rss = psutil.Process().memory_info().rss

    if gpu_available:
        console.print(
            f"  VRAM after load: {after_used_mb / 1024:.2f} / {after_total_mb / 1024:.2f} GB"
        )
    else:
        console.print("  VRAM: N/A (CPU-only mode)")

    console.print(f"  CPU RAM after load: {after_rss / 1024 / 1024:.2f} GB")

    # --- Compute peaks ---
    if gpu_available:
        vram_peak_mb = max(before_used_mb, after_used_mb)
        vram_peak_gb = round(vram_peak_mb / 1024 / 1024, 2)
    else:
        vram_peak_gb = 0.0

    cpu_ram_peak_gb = round(max(before_rss, after_rss) / 1024 / 1024, 2)

    console.print(
        f"  [bold green]VRAM peak: [/bold green] {vram_peak_gb} GB"
    )
    console.print(f"  [bold green]CPU RAM peak: [/bold green] {cpu_ram_peak_gb} GB")

    # --- Result ---
    result = {
        "vram_peak_gb": vram_peak_gb,
        "cpu_ram_peak_gb": cpu_ram_peak_gb,
        "model_file_size_gb": model_file_size_gb,
    }

    console.print("\n[bold green]Memory benchmark complete.[/bold green]")
    console.print(f"  VRAM peak:   {result['vram_peak_gb']} GB")
    console.print(f"  CPU RAM peak: {result['cpu_ram_peak_gb']} GB")
    console.print(f"  Model size:  {result['model_file_size_gb']} GB")

    return result
