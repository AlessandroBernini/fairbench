# FairBench

Fair benchmarking for local LLMs under 70B parameters on consumer hardware.

## Why FairBench

Most LLM benchmarks report results from datacenter GPUs (A100/H100) — useless for users running models locally on consumer hardware. FairBench measures **quality + speed + memory** on the hardware you actually own.

## Quick Start

```bash
# Install
pip install fairbench

# Run a benchmark
fairbench run --model qwen2.5-7b-instruct --quant q4_k_m

# View results
fairbench report --latest

# Publish to leaderboard (opt-out: results are public by default)
fairbench publish
```

## Features

- **Quality**: MMLU, HellaSwag, GSM8K (lite mode for speed)
- **Speed**: TTFT, TPOT, Throughput (tokens/sec)
- **Memory**: VRAM peak, CPU RAM peak
- **Hardware detection**: Auto-detect GPU/CPU, VRAM, RAM, OS
- **Privacy**: Opt-out publishing (results public by default)

## Architecture

```
fairbench/
├── fairbench/
│   ├── cli.py          # CLI entry point
│   ├── engine.py       # Benchmark orchestration
│   ├── backend.py      # llama.cpp backend
│   ├── reporter.py     # JSON reporter
│   ├── registry.py     # Model registry
│   └── config.py       # Configuration
├── tests/
├── pyproject.toml
└── README.md
```

## Contributing

PRs welcome! See `CONTRIBUTING.md` for guidelines.

## License

Apache 2.0
