# FairBench

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

# Skip publishing for this run only
fairbench run --model qwen2.5-7b-instruct --quant q4_k_m --no-publish

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

## Features

- **Quality**: MMLU, HellaSwag, GSM8K (lite mode for speed)
- **Speed**: TTFT, TPOT, Throughput (tokens/sec)
- **Memory**: VRAM peak, CPU RAM peak
- **Hardware detection**: Auto-detect GPU/CPU, VRAM, RAM, OS
- **Automatic publishing**: Results sent to leaderboard on every run (opt-out)

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
