Fair benchmarking for local LLMs under 70B parameters on consumer hardware.
## Why FairBench
Most LLM benchmarks report results from datacenter GPUs (A100/H100) — useless for users running models locally on consumer hardware. FairBench measures **quality + speed + memory** on the hardware you actually own.
## Quick Start
```bash
# Install
pip install fairbench

# Configure (optional) — publishing is ON by default
# Edit ~/.fairbench/config.yaml: publish: true/false

# Run a benchmark (published automatically to leaderboard)
fairbench run --model qwen2.5-7b-instruct --quant q4_k_m

# Run specific benchmarks
fairbench run --model qwen2.5-7b-instruct --tasks mmlu,humaneval,ifeval

# Skip publishing for this run only
fairbench run --model qwen2.5-7b-instruct --quant q4_k_m --no-publish

# View available benchmarks
fairbench list-benchmarks

# View results
fairbench report --latest
```

## Privacy

By default, every benchmark run is **automatically published** to the public leaderboard. This is an **opt-out** model — results are public unless you explicitly disable them.

To disable automatic publishing globally:
```yaml
# ~/.fairbench/config.yaml
publish: false
```

To skip publishing for a single run:
```bash
fairbench run -m <model> --no-publish
```

## Benchmarks

### Quality Benchmarks

| Benchmark | Description | Type | Lite Mode |
|-----------|-------------|------|-----------|
| **MMLU** | Knowledge across 57 academic subjects | Multiple choice | mmlu_pruning (~1000 questions) |
| **HellaSwag** | Commonsense reasoning & continuation | Multiple choice | Full dataset |
| **GSM8K** | Grade-school math reasoning | Exact match | 100 samples |
| **HumanEval** | Code generation (Python, 164 problems) | Pass@1 | 16 samples |
| **IFEval** | Instruction following compliance | Strict match | Subset |
| **TruthfulQA** | Truthfulness (knowledge + reasoning) | Multiple choice | Subset |
| **ARC-Challenge** | Scientific reasoning (Allen AI) | Multiple choice | Subset |
| **LiveBench** | 2024/2025 benchmark (anti contamination) | Multiple choice | Subset |

### Speed Benchmarks

- **TTFT** (Time To First Token): Latency in ms
- **TPOT** (Time Per Output Token): Latency per token in ms
- **Throughput**: Tokens per second

### Memory Benchmarks

- **VRAM Peak**: GPU memory usage during model load
- **CPU RAM Peak**: System memory usage during model load
- **Model File Size**: GGUF file size

## Usage Examples

```bash
# Full benchmark suite (default: MMLU, HellaSwag, GSM8K)
fairbench run -m qwen2.5-7b-instruct

# Custom benchmark set
fairbench run -m qwen2.5-7b-instruct --tasks humaneval,ifeval,truthfulqa

# Full (non-lite) mode for maximum accuracy
fairbench run -m qwen2.5-7b-instruct --full

# CPU-only mode
fairbench run -m qwen2.5-7b-instruct --n-gpu-layers 0

# Specify threads and token count
fairbench run -m qwen2.5-7b-instruct -t 8 -n 1024
```

## Architecture
```
fairbench/
├── fairbench/
│   ├── cli.py          # CLI entry point
│   ├── engine.py       # Benchmark orchestration
│   ├── backend.py      # Speed + memory benchmarking
│   ├── quality.py      # Quality benchmarking (lm_eval)
│   ├── reporter.py     # JSON reporter + leaderboard
│   ├── registry.py     # Model registry
│   └── config.py       # Configuration
├── api/                # Vercel API routes (TypeScript)
├── scripts/            # Setup + discovery scripts
├── tests/
├── pyproject.toml
└── README.md
```

## Roadmap

### ✅ Completed
- [x] Core benchmarking (quality, speed, memory)
- [x] 8 benchmark categories (MMLU, HellaSwag, GSM8K, HumanEval, IFEval, TruthfulQA, ARC, LiveBench)
- [x] Real TTFT/TPOT/throughput measurement via llama-cpp-python
- [x] Real VRAM/CPU RAM peak measurement via nvidia-smi + psutil
- [x] Vercel API + Neon DB + leaderboard frontend
- [x] Automatic publishing to public leaderboard

### 🔜 Next
- [ ] **SWE-bench integration** — Real-world software engineering benchmark (fix actual GitHub issues)
- [ ] Test suite (pytest)
- [ ] GitHub Actions CI workflow
- [ ] PyPI package publish
- [ ] Model discovery automation (weekly cron)

## Contributing
PRs welcome! See `CONTRIBUTING.md` for guidelines.
## License
Apache 2.0