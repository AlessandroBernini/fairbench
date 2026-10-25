"""Engine for FairBench benchmarks.

Orchestrates quality, speed, and memory benchmarks for a given model.
"""

from pathlib import Path
from typing import Optional

from fairbench.backend import detect_hardware, benchmark_speed, benchmark_memory
from fairbench.quality import run_quality_benchmark, DEFAULT_TASKS
from fairbench.reporter import Reporter
from fairbench.config import Config


def run_benchmark(
    model_path: Path,
    model_name: str,
    quantization: str = "q4_k_m",
    n_gpu_layers: int = -1,
    n_threads: int = 0,
    n_tokens: int = 512,
    lite: bool = True,
    publish: bool = True,
    tasks: list[str] | None = None,
) -> Path:
    """Run a complete benchmark on the specified model.

    Orchestrates quality, speed, and memory benchmarks. The quality
    benchmark runs on a configurable set of tasks (not just MMLU,
    HellaSwag, or GSM8K); when ``tasks`` is omitted, the default
    suite (MMLU, HellaSwag, GSM8K) is used for backward compatibility.

    Args:
        model_path: Path to the GGUF model file.
        model_name: Human-readable model name for the report.
        quantization: Quantization type (e.g., 'q4_k_m').
        n_gpu_layers: GPU layers to offload (-1 = all).
        n_threads: CPU threads (0 = auto).
        n_tokens: Number of tokens to generate for speed benchmark.
        lite: Use lite mode for quality benchmark (faster).
        publish: Whether to publish results to the leaderboard.
        tasks: Optional list of task names to run in the quality benchmark
            (e.g., ['mmlu', 'humaneval', 'ifeval']). Defaults to
            DEFAULT_TASKS when not provided.

    Returns:
        Path to the generated report file.
    """
    console = None
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        pass

    # Detect hardware
    if console:
        console.print("\n[bold green]=== FairBench Benchmark ===[/bold green]")
    hardware = detect_hardware()
    if console:
        console.print(f"\n[bold]Hardware:[/bold]")
        cpu = hardware['cpu']
        console.print(f"  CPU: {cpu.get('model', 'unknown')} ({cpu.get('cores', '?')} cores / {cpu.get('threads', '?')} threads)")
        console.print(f"  RAM: {hardware['ram_gb']} GB")
        gpu = hardware['gpu']
        if gpu['type'] != 'none':
            console.print(f"  GPU: {gpu['model']} ({gpu['vram_gb']} GB VRAM)")
        else:
            console.print("  GPU: None (CPU-only mode)")
        console.print(f"  OS: {hardware['os']}")

    # Default tasks for backward compatibility
    if tasks is None:
        tasks = DEFAULT_TASKS

    # 1. Quality benchmark
    if console:
        console.print("\n" + "=" * 60)
    quality = run_quality_benchmark(
        model_path=model_path,
        task_names_in=tasks,
        lite=lite,
        batch_size=4,
    )
    if console:
        console.print("\n[bold green]Quality complete.[/bold green]")

    # 2. Speed benchmark
    if console:
        console.print("\n" + "=" * 60)
    speed = benchmark_speed(
        model_path=model_path,
        prompt="Hello, how are you today?",
        n_tokens=n_tokens,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
    )
    if console:
        console.print("\n[bold green]Speed complete.[/bold green]")

    # 3. Memory benchmark
    if console:
        console.print("\n" + "=" * 60)
    memory = benchmark_memory(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )
    if console:
        console.print("\n[bold green]Memory complete.[/bold green]")

    # Generate report
    config = Config()
    reporter = Reporter(config.results_dir)
    report_path = reporter.generate_report(
        model_name=model_name,
        quantization=quantization,
        hardware=hardware,
        quality=quality,
        speed=speed,
        memory=memory,
        publish=publish,
    )

    if console:
        console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")
        if publish:
            console.print(f"[bold green]Report published to leaderboard[/bold green]")
        else:
            console.print(f"[bold yellow]Report saved locally (not published)[/bold yellow]")

    return report_path
