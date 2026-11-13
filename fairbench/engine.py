"""Engine orchestration for FairBench."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fairbench.backend import benchmark_memory, benchmark_speed
from fairbench.config import load_config
from fairbench.data import DataManager
from fairbench.quality import AVAILABLE_BENCHMARKS, get_lite_benchmarks, run_quality_benchmark
from fairbench.reporter import Reporter
from fairbench.swebench import SWEbenchClient
from fairbench.types import (
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkRun,
    LiteMode,
)

logger = logging.getLogger("fairbench")


class BenchmarkEngine:
    """Orchestrates benchmark runs."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or load_config()
        self.data = DataManager(self.config.results_dir)
        self.reporter = Reporter(self.config)
        self.swebench_client = SWEbenchClient(self.data, self.config)

    def run_benchmark(
        self,
        model_path: str,
        category: BenchmarkCategory = BenchmarkCategory.QUALITY,
        tasks: list[str] | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        quantization: str | None = None,
        lite_mode: str | None = None,
    ) -> BenchmarkRun:
        """Run a complete benchmark suite or specific category.

        Args:
            model_path: Path to GGUF model file
            category: Which benchmarks to run
            tasks: Specific quality benchmarks to run
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of CPU threads
            quantization: Quantization type
            lite_mode: Lite mode setting
        """
        n_ctx = n_ctx or self.config.default_n_ctx
        n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else self.config.default_n_gpu_layers
        )
        n_threads = n_threads if n_threads is not None else self.config.default_threads
        quantization = quantization or self.config.default_quant
        lite_mode_val = lite_mode or "auto"

        logger.info("FairBench v%s", "0.3.0")
        logger.info("Model: %s", model_path)
        logger.info("Category: %s", category.value)
        logger.info("Context: %d | GPU Layers: %d | Threads: %d", n_ctx, n_gpu_layers, n_threads)
        logger.info("Quantization: %s", quantization)

        # Verify model file exists
        if not Path(model_path).exists():
            logger.error("Error: Model file not found: %s", model_path)
            logger.info("Possible fixes:")
            logger.info("  • Check the path is correct")
            logger.info("  • Use an absolute path (e.g., ~/models/model.gguf)")
            logger.info("  • Download a model first:")
            logger.info("    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --include '*.gguf*' --local-dir ~/models/")
            raise SystemExit(1)

        run = BenchmarkRun(
            model_name=Path(model_path).stem,
            model_path=model_path,
            quantization=quantization,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
        )

        # Run specified category
        run_id = f"checkpoint_{int(time.time())}"
        if category in (
            BenchmarkCategory.QUALITY,
            BenchmarkCategory.SPEED,
            BenchmarkCategory.MEMORY,
        ):
            run = self._run_quality_speed_memory(
                run, model_path, tasks, n_ctx, n_gpu_layers, n_threads, lite_mode_val, run_id
            )

        if category == BenchmarkCategory.SWE_BENCH:
            run = self._run_swebench(run, model_path, lite_mode_val)

        result_dict = {
            "run_id": run_id,
            "model_name": run.model_name,
            "model_path": run.model_path,
            "quantization": run.quantization,
            "n_ctx": run.n_ctx,
            "n_gpu_layers": run.n_gpu_layers,
            "n_threads": run.n_threads,
            "timestamp": run.timestamp,
            "category": category.value,
            "quality_results": [
                {
                    "benchmark_name": r.benchmark_name,
                    "score": r.score,
                    "items_total": r.items_total,
                    "items_correct": r.items_correct,
                    "metadata": r.metadata,
                }
                for r in run.quality_results
            ],
            "speed_result": {
                "ttft_avg": run.speed_result.ttft_avg if run.speed_result else 0,
                "tpot_avg": run.speed_result.tpot_avg if run.speed_result else 0,
                "throughput_avg": run.speed_result.throughput_avg if run.speed_result else 0,
                "runs": run.speed_result.runs if run.speed_result else 0,
                "metadata": run.speed_result.metadata if run.speed_result else {},
            }
            if run.speed_result
            else None,
            "memory_result": {
                "vram_peak_mb": run.memory_result.vram_peak_mb if run.memory_result else 0,
                "ram_peak_mb": run.memory_result.ram_peak_mb if run.memory_result else 0,
                "metadata": run.memory_result.metadata if run.memory_result else {},
            }
            if run.memory_result
            else None,
            "swebench_results": [
                {
                    "instance_id": r.instance_id,
                    "repo_name": r.repo_name,
                    "resolved": r.resolved,
                    "test_results": r.test_results,
                    "metadata": r.metadata,
                }
                for r in run.swebench_results
            ],
        }

        # Attach hardware profile to final result
        from fairbench.hardware_profile import get_hardware_profile

        hw_profile = get_hardware_profile()
        result_dict["hardware_profile_hash"] = hw_profile.profile_hash
        result_dict["hardware_snapshot"] = hw_profile.to_dict()

        saved_path = self.data.save_result(run_id, result_dict)
        logger.info("Final result saved to: %s", saved_path)

        # Publish if configured
        if self.config.publish:
            result = self.reporter.publish_result(run_id, result_dict)
            if result:
                logger.info("Result published to leaderboard")
            else:
                logger.warning("Failed to publish result (API unavailable)")
        else:
            logger.info("Result not published (publish disabled in config)")

        return run

    def _run_quality_speed_memory(
        self,
        run: BenchmarkRun,
        model_path: str,
        tasks: list[str] | None,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        lite_mode: str,
        run_id: str,
    ) -> BenchmarkRun:
        """Run quality, speed, and memory benchmarks."""
        # Convert string to LiteMode enum
        mode_map = {
            "enabled": LiteMode.ENABLED,
            "disabled": LiteMode.DISABLED,
            "auto": LiteMode.AUTO,
        }
        lite_mode_enum = mode_map.get(lite_mode, LiteMode.AUTO)

        # Quality
        logger.info("[Quality Benchmark]")
        try:
            quality_results = run_quality_benchmark(
                model_path=model_path,
                tasks=tasks,
                lite_mode=lite_mode_enum,
                config=self.config,
                run_id=run_id,
            )
            run.quality_results = quality_results

            for result in quality_results:
                logger.info(
                    "  %s: %.4f (%d/%d)",
                    result.benchmark_name,
                    result.score,
                    result.items_correct,
                    result.items_total,
                )

            # Save partial results after quality
            self._save_partial_result(run_id, run, category="quality")
        except ImportError as e:
            if "lm_eval" in str(e).lower():
                logger.error("lm-eval not installed")
                logger.info("Quality benchmarks require lm-eval")
                logger.info("Install with:")
                logger.info("   pip install fairbench-ai[ml]")
            else:
                raise

        # Speed
        logger.info("[Speed Benchmark]")
        try:
            speed_result = benchmark_speed(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
            )
            run.speed_result = speed_result
            logger.info("  TTFT: %.2fms", speed_result.ttft_avg)
            logger.info("  TPOT: %.4f tokens/s", speed_result.tpot_avg)
            logger.info("  Throughput: %.2f tokens/s", speed_result.throughput_avg)

            # Save partial results after speed
            self._save_partial_result(run_id, run, category="speed")
        except ImportError as e:
            if "llama_cpp" in str(e).lower() or "llama-cpp" in str(e).lower():
                logger.error("llama-cpp-python not installed")
                logger.info("Speed/memory benchmarks require llama-cpp-python")
                logger.info("Install with:")
                logger.info("   pip install llama-cpp-python")
            else:
                raise

        # Memory
        logger.info("[Memory Benchmark]")
        try:
            memory_result = benchmark_memory(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
            )
            run.memory_result = memory_result
            logger.info("  VRAM Peak: %.0f MB", memory_result.vram_peak_mb)
            logger.info("  RAM Peak: %.0f MB", memory_result.ram_peak_mb)

            # Save partial results after memory
            self._save_partial_result(run_id, run, category="memory")
        except ImportError as e:
            if "llama_cpp" in str(e).lower() or "llama-cpp" in str(e).lower():
                logger.error("llama-cpp-python not installed")
                logger.info("Speed/memory benchmarks require llama-cpp-python")
                logger.info("Install with:")
                logger.info("   pip install llama-cpp-python")
            else:
                raise

        return run

    def _save_partial_result(
        self,
        run_id: str,
        run: BenchmarkRun,
        category: str,
    ) -> None:
        """Save partial results for a category."""
        result_dict = {
            "run_id": run_id,
            "model_name": run.model_name,
            "model_path": run.model_path,
            "quantization": run.quantization,
            "n_ctx": run.n_ctx,
            "n_gpu_layers": run.n_gpu_layers,
            "n_threads": run.n_threads,
            "timestamp": run.timestamp,
            "category": category,
            "quality_results": [
                {
                    "benchmark_name": r.benchmark_name,
                    "score": r.score,
                    "items_total": r.items_total,
                    "items_correct": r.items_correct,
                    "metadata": r.metadata,
                }
                for r in run.quality_results
            ],
            "speed_result": {
                "ttft_avg": run.speed_result.ttft_avg if run.speed_result else 0,
                "tpot_avg": run.speed_result.tpot_avg if run.speed_result else 0,
                "throughput_avg": run.speed_result.throughput_avg if run.speed_result else 0,
                "runs": run.speed_result.runs if run.speed_result else 0,
                "metadata": run.speed_result.metadata if run.speed_result else {},
            }
            if run.speed_result
            else None,
            "memory_result": {
                "vram_peak_mb": run.memory_result.vram_peak_mb if run.memory_result else 0,
                "ram_peak_mb": run.memory_result.ram_peak_mb if run.memory_result else 0,
                "metadata": run.memory_result.metadata if run.memory_result else {},
            }
            if run.memory_result
            else None,
        }
        self.data.save_result(run_id, result_dict)
        logger.info("Partial result saved for '%s'", category)

    def _run_swebench(
        self,
        run: BenchmarkRun,
        model_path: str,
        lite_mode: str,
    ) -> BenchmarkRun:
        """Run SWE-bench evaluation."""
        split = "lite" if lite_mode == "enabled" else "full"

        logger.info("[SWE-bench Evaluation]")
        results = self.swebench_client.run_evaluation(
            model_path=model_path,
            split=split,
        )
        run.swebench_results = results

        stats = self.swebench_client.compute_stats(results)
        logger.info("  Total: %d", stats.total_instances)
        logger.info("  Resolved: %d", stats.resolved)
        logger.info("  Failed: %d", stats.failed)
        logger.info("  Errors: %d", stats.errors)
        logger.info("  Score: %.1f%%", stats.avg_test_score)

        return run

    def list_available_benchmarks(self) -> dict:
        """List all available quality benchmarks."""
        return AVAILABLE_BENCHMARKS

    def get_lite_benchmarks(self) -> list[str]:
        """Get list of benchmarks that support lite mode."""
        return get_lite_benchmarks()

    def list_swebench_splits(self) -> dict:
        """List available SWE-bench splits."""
        from fairbench.swebench import SWEBENCH_SOURCES

        return SWEBENCH_SOURCES

    def get_swebench_instances(
        self,
        split: str = "full",
        limit: int | None = None,
    ) -> list[dict]:
        """Get available SWE-bench instances."""
        instances = self.swebench_client.fetch_instances(split=split, limit=limit)
        return [
            {
                "instance_id": inst.instance_id,
                "repo": inst.repo,
                "code_language": inst.code_language,
            }
            for inst in instances
        ]
