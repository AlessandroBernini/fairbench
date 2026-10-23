"""Engine for FairBench benchmarks."""

from pathlib import Path
from typing import Optional

from fairbench.backend import detect_hardware, benchmark_speed, benchmark_memory
from fairbench.reporter import Reporter
from fairbench.config import Config


def run_benchmark(
    model_path: Path,
    model_name: str,
    quantization: str = "q4_k_m",
    n_gpu_layers: int = -1,
    n_threads: int = 0,
    n_tokens: int = 512,
    publish: bool = True,
) -> Path:
    """Run a complete benchmark."""
    # Detect hardware
    hardware = detect_hardware()

    # Run speed benchmark
    speed = benchmark_speed(
        model_path=model_path,
        prompt="Hello, how are you?",
        n_tokens=n_tokens,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
    )

    # Run memory benchmark
    memory = benchmark_memory(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )

    # Run quality benchmark (placeholder)
    quality = {
        "mmlu": 0.0,
        "hellaswag": 0.0,
        "gsm8k": 0.0,
    }

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

    return report_path
