"""Quality benchmarking for AI models."""

from __future__ import annotations

import time

from fairbench.types import BenchmarkConfig, LiteMode, QualityResult

AVAILABLE_BENCHMARKS = {
    "mmlu": {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding",
        "subjects": 57,
        "lite": True,
    },
    "hellaswag": {
        "name": "HellaSwag",
        "description": "Commonsense reasoning",
        "lite": True,
    },
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade school math word problems",
        "lite": True,
    },
    "humaneval": {
        "name": "HumanEval",
        "description": "Code generation (Pass@1)",
        "lite": False,
    },
    "ifeval": {
        "name": "IFEval",
        "description": "Instruction following",
        "lite": True,
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "description": "Truthfulness evaluation",
        "lite": True,
    },
    "arc_challenge": {
        "name": "ARC-Challenge",
        "description": "Scientific reasoning",
        "lite": True,
    },
    "livebench": {
        "name": "LiveBench",
        "description": "2024/2025 anti-contamination benchmark",
        "lite": True,
    },
}


def list_benchmarks() -> dict[str, dict]:
    """Return available benchmarks with metadata."""
    return dict(AVAILABLE_BENCHMARKS)


def get_lite_benchmarks() -> list[str]:
    """Return list of benchmarks that support lite mode."""
    return [name for name, info in AVAILABLE_BENCHMARKS.items() if info["lite"]]


def run_quality_benchmark(
    model_path: str,
    benchmark_name: str | None = None,
    tasks: list[str] | None = None,
    lite_mode: LiteMode = LiteMode.AUTO,
    config: BenchmarkConfig | None = None,
) -> list[QualityResult]:
    """Run quality benchmarks on a model.

    Falls back to zero scores if lm_eval is not installed.
    """
    # Check if lm_eval is available
    try:
        from lm_eval import simple_evaluate  # type: ignore

        has_lm_eval = True
    except ImportError:
        has_lm_eval = False

    if not has_lm_eval:
        # Check if we're in a non-interactive environment (CI, cron, etc.)
        is_interactive = hasattr(__import__('sys').stdin, 'isatty') and __import__('sys').stdin.isatty()

        benchmarks_to_run = (
            [benchmark_name] if benchmark_name else list(AVAILABLE_BENCHMARKS.keys())
        )
        if tasks:
            benchmarks_to_run = tasks

        if is_interactive:
            print("\n⚠ lm_eval not found — quality benchmarks require it.")
            response = input("   Install lm-eval now? [y/N]: ").strip().lower()
            if response in ('y', 'yes'):
                print("\n   Installing lm-eval...")
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "lm-eval"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("   ✓ lm-eval installed successfully.")
                    # Retry import
                    try:
                        from lm_eval import simple_evaluate  # type: ignore
                        has_lm_eval = True
                    except ImportError:
                        print("   ✗ Still can't import lm-eval after install. Skipping benchmarks.\n")
            else:
                print("   Skipping quality benchmarks (lm_eval not installed).\n")

        if not has_lm_eval:
            # Fallback to zero scores
            results = []
            for name in benchmarks_to_run:
                info = AVAILABLE_BENCHMARKS.get(name, {})
                results.append(
                    QualityResult(
                        benchmark_name=name,
                        score=0.0,
                        items_total=0,
                        items_correct=0,
                        metadata={"reason": "lm_eval not installed", "fallback": True},
                    )
                )
            return results
        # If user said yes and install succeeded, continue to benchmark execution below

    # Determine which benchmarks to run
    benchmarks_to_run = [benchmark_name] if benchmark_name else list(AVAILABLE_BENCHMARKS.keys())
    if tasks:
        benchmarks_to_run = tasks

    # Filter by lite mode
    if lite_mode == LiteMode.ENABLED:
        lite_benchmarks = get_lite_benchmarks()
        benchmarks_to_run = [b for b in benchmarks_to_run if b in lite_benchmarks]
    elif lite_mode == LiteMode.DISABLED:
        pass  # Run all
    else:  # AUTO
        pass  # No filtering; run all available benchmarks

    results = []
    for benchmark in benchmarks_to_run:
        info = AVAILABLE_BENCHMARKS.get(benchmark, {})
        print(f"  Running {info.get('name', benchmark)}...")

        start = time.time()
        try:
            eval_results = simple_evaluate(
                model="llama-cpp",
                model_args=model_path,
                tasks=[benchmark],
                batch_size="auto",
                log_samples=True,
                verbosity="WARNING",
            )

            score = eval_results.get("results", {}).get(benchmark, {}).get("acc", 0.0)
            items_total = eval_results.get("results", {}).get(benchmark, {}).get("items", 0)
            items_correct = eval_results.get("results", {}).get(benchmark, {}).get("correct", 0)

            elapsed = time.time() - start
            results.append(
                QualityResult(
                    benchmark_name=benchmark,
                    score=score,
                    items_total=items_total,
                    items_correct=items_correct,
                    metadata={
                        "elapsed_seconds": elapsed,
                        "description": info.get("description", ""),
                    },
                )
            )
            print(f"    Score: {score:.4f} ({items_correct}/{items_total}) in {elapsed:.1f}s")
        except (OSError, RuntimeError) as e:
            print(f"    Error: {e}")
            results.append(
                QualityResult(
                    benchmark_name=benchmark,
                    score=0.0,
                    items_total=0,
                    items_correct=0,
                    metadata={"error": str(e)},
                )
            )

    return results
