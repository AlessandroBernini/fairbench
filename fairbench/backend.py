"""Speed and memory benchmarking for AI models."""

from __future__ import annotations

import os
import subprocess
import time

import psutil
from rich.progress import Progress, SpinnerColumn, TextColumn

from fairbench.types import MemoryResult, SpeedResult


def benchmark_speed(
    model_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 10,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: int = 0,
    runs: int = 3,
) -> SpeedResult:
    """Measure TTFT, TPOT, and throughput using llama-cpp-python bindings.

    Falls back to llama-bench CLI if bindings unavailable.
    """
    try:
        from llama_cpp import Llama as _Llama  # type: ignore
    except ImportError:
        return _benchmark_with_llama_bench(model_path, prompt, runs)

    return _benchmark_with_llama_cpp(
        model_path, prompt, n_ctx, n_gpu_layers, n_threads, runs, _Llama
    )


def _benchmark_with_llama_cpp(
    model_path: str,
    prompt: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    runs: int,
    Llama: type,  # type: ignore[type-arg]
) -> SpeedResult:
    """Benchmark using llama-cpp-python directly."""
    ttfts: list[float] = []
    tpots: list[float] = []
    throughputs: list[float] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("  [bold blue]Speed[/]  {task.description}", justify="right"),
        refresh_per_second=1,
    ) as progress:
        for i in range(runs):
            task = progress.add_task(f"Run {i + 1}/{runs}", total=None)
            try:
                llm = Llama(  # type: ignore[no-untyped-call]
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads if n_threads > 0 else None,
                    verbose=False,
                )

                start = time.time()
                response = llm.create_completion(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stream=True,
                )

                ttft = time.time() - start
                ttfts.append(ttft * 1000)  # ms

                token_count = 0
                last_time = time.time()
                for chunk in response:
                    if time.time() - last_time > 0.01:
                        elapsed = time.time() - last_time
                        tpots.append((token_count / elapsed) if elapsed > 0 else 0)
                        token_count = 0
                        last_time = time.time()
                    token_count += 1

                total_time = time.time() - start
                throughput = token_count / total_time if total_time > 0 else 0
                throughputs.append(throughput)

                llm.close()
                progress.update(task, completed=1)
            except (OSError, RuntimeError) as e:
                progress.update(task, completed=1)
                print(f"    Run {i + 1} failed: {e}")

    return SpeedResult(
        ttft_avg=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        tpot_avg=sum(tpots) / len(tpots) if tpots else 0.0,
        throughput_avg=sum(throughputs) / len(throughputs) if throughputs else 0.0,
        runs=len(ttfts),
    )


def _benchmark_with_llama_bench(model_path: str, prompt: str, runs: int) -> SpeedResult:
    """Fallback: use llama-bench CLI tool."""
    try:
        cmd = [
            "llama-bench",
            "-m",
            model_path,
            "-n",
            "512",
            "-ngl",
            "0",
            "-p",
            str(len(prompt.split())),
            "-t",
            "0",
            "-r",
            str(runs),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
        # Parse output - llama-bench format varies by version
        output = result.stdout
        return SpeedResult(runs=runs, metadata={"raw_output": output})
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return SpeedResult(runs=runs, metadata={"error": str(e)})


def benchmark_memory(
    model_path: str,
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 10,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: int = 0,
) -> MemoryResult:
    """Measure VRAM and RAM peak usage during inference."""
    try:
        from llama_cpp import Llama as _Llama  # type: ignore
    except ImportError:
        return MemoryResult(metadata={"reason": "llama-cpp-python not installed"})

    return _measure_memory_llama_cpp(
        model_path,
        prompt,
        n_ctx,
        n_gpu_layers,
        n_threads,
        _Llama,  # type: ignore[arg-type]
    )


def _measure_memory_llama_cpp(
    model_path: str,
    prompt: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    Llama: type,  # type: ignore[type-arg]
) -> MemoryResult:
    """Measure memory using llama-cpp-python with peak tracking."""
    vram_samples: list[float] = []
    ram_samples: list[float] = []

    try:
        # Get initial memory state
        initial_ram = psutil.Process().memory_info().rss

        llm = Llama(  # type: ignore[no-untyped-call]
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads if n_threads > 0 else None,
            verbose=False,
        )

        # Measure VRAM via nvidia-smi if available
        if os.path.exists("/usr/bin/nvidia-smi"):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                vram_samples = [
                    float(line.strip())
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]

        # Run inference
        llm.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
        )

        # Peak RAM
        current_ram = psutil.Process().memory_info().rss
        ram_peak = (current_ram - initial_ram) / (1024 * 1024)  # MB
        ram_samples.append(ram_peak)

        llm.close()

        return MemoryResult(
            vram_peak_mb=sum(vram_samples) / len(vram_samples) if vram_samples else 0.0,
            ram_peak_mb=ram_peak,
            vram_avg_mb=sum(vram_samples) / len(vram_samples) if vram_samples else 0.0,
            ram_avg_mb=sum(ram_samples) / len(ram_samples) if ram_samples else 0.0,
            metadata={"model_size_gb": os.path.getsize(model_path) / (1024**3)},
        )
    except (OSError, RuntimeError) as e:
        return MemoryResult(metadata={"error": str(e)})
