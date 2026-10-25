"""Type definitions for FairBench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BenchmarkCategory(Enum):
    QUALITY = "quality"
    SPEED = "speed"
    MEMORY = "memory"
    SWE_BENCH = "swebench"


class QuantizationType(Enum):
    Q4_K_M = "q4_k_m"
    Q3_K_M = "q3_k_m"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    F16 = "f16"
    F32 = "f32"


class LiteMode(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"


@dataclass
class QualityResult:
    """Results from quality benchmarks."""
    benchmark_name: str
    score: float
    items_total: int
    items_correct: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeedResult:
    """Results from speed benchmarks."""
    ttft_avg: float = 0.0  # Time to first token (ms)
    tpot_avg: float = 0.0  # Tokens per second
    throughput_avg: float = 0.0
    runs: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResult:
    """Results from memory benchmarks."""
    vram_peak_mb: float = 0.0
    ram_peak_mb: float = 0.0
    vram_avg_mb: float = 0.0
    ram_avg_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SWEbenchResult:
    """Results from SWE-bench evaluation."""
    repo_name: str
    instance_id: str
    resolved: bool
    test_results: dict[str, Any] = field(default_factory=dict)
    patch_applied: bool = False
    patch_success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    model_name: str
    model_path: str
    quantization: str
    n_ctx: int
    n_gpu_layers: int
    n_threads: int
    timestamp: str
    category: BenchmarkCategory
    quality_results: list[QualityResult] = field(default_factory=list)
    speed_result: SpeedResult | None = None
    memory_result: MemoryResult | None = None
    swebench_results: list[SWEbenchResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Model metadata."""
    name: str
    path: str
    quantization: str
    size_gb: float
    context_length: int
    parameters: int
    architecture: str = ""
    license: str = ""


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    api_url: str = "https://fairbench.vercel.app/api"
    default_n_ctx: int = 4096
    default_n_gpu_layers: int = -1
    default_quant: str = "q4_k_m"
    default_threads: int = 0
    publish: bool = True
    results_dir: str = "/home/jarvispi/.fairbench/results"
    lite_mode: LiteMode = LiteMode.AUTO
    swebench_instances: int = 0
    swebench_split: str = "full"
