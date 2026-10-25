# FairBench

AI model benchmarking suite with SWE-bench integration.

## Quick Start

```bash
pip install fair-bench
fairbench run -m model.gguf
```

## Benchmarks

- **Quality**: MMLU, HellaSwag, GSM8K, HumanEval, IFEval, TruthfulQA, ARC-Challenge, LiveBench
- **Speed**: TTFT, TPOT, throughput (llama-cpp-python)
- **Memory**: VRAM peak, RAM peak
- **SWE-bench**: GitHub issue resolution capability

## Configuration

Edit `~/.fairbench/config.yaml`:

```yaml
api_url: https://fairbench.vercel.app/api
default_n_ctx: 4096
default_n_gpu_layers: -1
default_quant: q4_k_m
default_threads: 0
publish: true
results_dir: /home/jarvispi/.fairbench/results
```

## Publishing

By default, every benchmark run is **automatically published** to the public leaderboard. Set `publish: false` to disable.
