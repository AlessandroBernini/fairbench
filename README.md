# FairBench

AI model benchmarking suite with SWE-bench integration.

Evaluate AI models across multiple dimensions: quality, speed, memory usage, and software engineering capability.

## Quick Start

```bash
pip install fairbench-ai[ml]
fairbench run ~/models/qwen3-35b-q4_k_m.gguf
```

> **Prerequisito:** `fairbench-ai[ml]` include `llama-cpp-python` (necessario per speed/memory benchmarks). Su Apple Silicon serve Xcode command-line tools (`xcode-select --install`).

## Benchmarks

### Quality

FairBench evaluates models on **8 quality benchmarks** (7 lite, 1 full):

| Benchmark | Description | Lite | Subjects/Tasks |
|-----------|-------------|------|----------------|
| **MMLU** | Massive Multitask Language Understanding | Yes | 57 subjects |
| **HellaSwag** | Commonsense reasoning | Yes | 10,000+ tasks |
| **GSM8K** | Grade school math word problems | Yes | 8,500+ problems |
| **IFEval** | Instruction following | Yes | 2,400+ instances |
| **TruthfulQA** | Truthfulness evaluation | Yes | 800+ questions |
| **ARC-Challenge** | Scientific reasoning | Yes | 2,590 questions |
| **LiveBench** | 2024/2025 anti-contamination benchmark | Yes | Multiple tasks |
| **HumanEval** | Code generation (Pass@1) | No | 164 tasks |

### Speed

- **TTFT** (Time to First Token) — latency
- **TPOT** (Tokens Per Second) — throughput
- Measured via `llama-cpp-python`

### Memory

- **VRAM peak** — GPU memory usage
- **RAM peak** — CPU memory usage

### SWE-bench

- Evaluate models on **GitHub issue resolution** tasks
- Splits: **full** (all issues), **lite** (10% stratified sample), **verified** (human-verified only)
- Fetches real issues from the SWE-bench dataset

## Usage

### Run benchmarks

```bash
# All categories (quality + speed + memory)
fairbench run ~/models/model.gguf --category all

# Quality only with specific benchmarks
fairbench run ~/models/model.gguf --category quality --tasks mmlu hellaswag

# Lite mode (faster, smaller benchmark subsets)
fairbench run ~/models/model.gguf --category all --lite

# SWE-bench evaluation
fairbench run ~/models/model.gguf --category swebench --swebench-split lite

# SWE-bench with limit
fairbench run ~/models/model.gguf --category swebench --swebench-split verified -l 10

# Custom context window and GPU layers
fairbench run ~/models/model.gguf --n-ctx 8192 --n-gpu-layers 30
```

### List available benchmarks

```bash
fairbench list-benchmarks
fairbench list-swebench
```

### View results

```bash
fairbench results
```

### View leaderboard

```bash
fairbench leaderboard
```

### Manage configuration

```bash
fairbench config --show       # Show current config
fairbench config --set key=value  # Set a config value
```

## Configuration

Edit `~/.fairbench/config.yaml`:

```yaml
api_url: https://fairbench.vercel.app/api
default_n_ctx: 4096
default_n_gpu_layers: -1
default_quant: q4_k_m
default_threads: 0
publish: true
results_dir: ~/.fairbench/results
```

| Key | Default | Description |
|-----|---------|-------------|
| `api_url` | `https://fairbench.vercel.app/api` | Leaderboard API endpoint |
| `default_n_ctx` | `4096` | Default context window size |
| `default_n_gpu_layers` | `-1` | GPU offload layers (-1 = all) |
| `default_quant` | `q4_k_m` | Default quantization type |
| `default_threads` | `0` | CPU threads (0 = auto) |
| `publish` | `true` | Auto-publish results to leaderboard |
| `results_dir` | `~/.fairbench/results` | Local results storage |

## Publishing

By default, every benchmark run is **automatically published** to the public leaderboard. Set `publish: false` in config to disable.

## Optional Dependencies

```bash
# ML benchmarks (lm-eval, llama-cpp-python)
pip install fairbench-ai[ml]

# SWE-bench evaluation
pip install fairbench-ai[swebench]

# Development
pip install fairbench-ai[dev]
```

## Requirements

- Python 3.10+
- [llama-cpp-python](https://github.com/ggerganov/llama.cpp) for speed/memory benchmarks
