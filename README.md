# FairBench — AI model benchmarking suite with SWE-bench integration.

Evaluate AI models across multiple dimensions: quality, speed, memory usage,
and software engineering capability.

## Quick Start

```bash
# Install the core CLI (results, config, leaderboard tooling)
pip install fairbench-ai

# Full ML stack for quality benchmarks and SWE-bench evaluation
pip install "fairbench-ai[ml]"

# Speed/memory benchmarks need llama-cpp-python (compiled; see Optional Dependencies)
pip install llama-cpp-python

# Create default config (~/.fairbench/config.yaml)
fairbench init

# ...or run the interactive setup wizard
fairbench init --wizard

# Run benchmarks on a GGUF model
fairbench run ~/models/qwen3-35b-q4_k_m.gguf
```

> **Note:** speed/memory benchmarks require `llama-cpp-python`. On Apple Silicon, Xcode command-line tools are required (`xcode-select --install`). Auto-version bumping is enabled on push to main.

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

Requires the `[ml]` extra (`lm-eval`). If `lm-eval` is not installed, FairBench asks whether to install it interactively; in non-interactive environments (CI/cron) the quality benchmark is skipped with a warning.

### Speed

Measured via `llama-cpp-python` bindings over 3 averaged runs (falls back to the `llama-bench` CLI if bindings are unavailable):

- **TTFT** (Time To First Token) — first-token latency, in ms
- **TPOT** (Time Per Output Token) — mean decode time per token, in ms, excluding the first generated token
- **Throughput** — generated tokens / decode time, in tokens/s

### Memory

- **VRAM peak** — GPU memory usage (via `nvidia-smi` when available)
- **RAM peak** — CPU memory usage (via `psutil`)

Works without extra dependencies.

### SWE-bench

- Evaluate models on **GitHub issue resolution** tasks
- Splits: **full** (all issues), **lite** (10% stratified sample), **verified** (human-verified only)
- Fetches real issues from the SWE-bench dataset (requires `[ml]` extra)

## Usage

### Run benchmarks

```bash
# All categories (quality + speed + memory)
fairbench run ~/models/model.gguf --category all

# Quality only with specific benchmarks (--tasks is repeatable, NOT comma-separated)
fairbench run ~/models/model.gguf --category quality --tasks mmlu --tasks hellaswag

# Lite mode (faster, smaller benchmark subsets)
fairbench run ~/models/model.gguf --category all --lite

# SWE-bench evaluation
fairbench run ~/models/model.gguf --category swebench --swebench-split lite

# SWE-bench with limit
fairbench run ~/models/model.gguf --category swebench --swebench-split verified -l 10

# Custom context window and GPU layers
fairbench run ~/models/model.gguf --n-ctx 8192 --n-gpu-layers 30

# CPU threads and quantization
fairbench run ~/models/model.gguf --n-threads 8 --quant q5_k_m

# Category shorthand, repeated task shorthand, lite flag
fairbench run model.gguf -k quality -t mmlu -t ifeval --lite

# Force or skip leaderboard publishing for this run only
fairbench run model.gguf --publish
fairbench run model.gguf --no-publish

# Run inside Docker container (builds/updates the image, layer-cached)
fairbench run --docker ~/models/model.gguf

# Run with verbose logging and file output (-v INFO, -vv DEBUG)
fairbench run ~/models/model.gguf -vv --log-file /tmp/fairbench.log
```

#### `run` options (as implemented)

| Flag | Shorthand | Default | Description |
|------|-----------|---------|-------------|
| `MODEL_PATH` | — | required | Path to a GGUF file, or a catalog name (`fairbench models list`) |
| `--category` | `-k` | `all` | `all`, `quality`, `speed`, `memory`, `swebench` |
| `--tasks` | `-t` | all lite-compatible | Repeatable: specific quality benchmarks (e.g. `-t mmlu -t ifeval`) |
| `--n-ctx` | `-n` | config `default_n_ctx` | Context window size |
| `--n-gpu-layers` | `-g` | config `default_n_gpu_layers` | GPU offload layers (-1 = all) |
| `--n-threads` | `-j` | config `default_threads` | CPU threads (0 = auto) |
| `--quant` | `-q` | config `default_quant` | Quantization label |
| `--lite` | — | off | Use lite benchmark subsets |
| `--swebench-split` | `-s` | `full` | `full`, `lite`, or `verified` |
| `--swebench-limit` | `-l` | config `swebench_instances` | Max SWE-bench instances |
| `--publish` / `--no-publish` | — | from config | Override leaderboard publishing for this run |
| `--verbose` | `-v`, `-vv` | WARNING | INFO with `-v`, DEBUG with `-vv` |
| `--log-file PATH` | — | none | Structured JSON log file |
| `--docker` | — | off | Run the benchmark inside a Docker container |
| `--docker-image NAME` | — | `fairbench:latest` | Image used for `--docker` runs |
| `--ci` | — | off | CI mode (no TTY, no progress bars) |

### SWE-bench dedicated command

```bash
# Run SWE-bench evaluation directly (not via 'run')
fairbench swebench-eval ~/models/model.gguf --split lite --limit 10
fairbench swebench-eval model.gguf -s verified -r flask --limit 5
```

`--split/-s` default `full`, `--limit/-l` default `5`, `--repo-filter/-r` filters by repository.

### List available benchmarks and instances

```bash
fairbench list-benchmarks
fairbench list-swebench                  # default: split=full, limit=10
fairbench list-swebench --split verified -l 20
```

## Command reference

| Command | Description |
|---------|-------------|
| `run MODEL` | Run benchmarks (see options above) |
| `interactive` | Guided interactive benchmark configuration and run |
| `quick-start` | Demo plan with available benchmarks and models |
| `results [--category/-c]` | List local benchmark results |
| `compare [-r runs] [-m model] [-c category] [-f console\|csv\|json] [-s sort-by]` | Compare results across models/runs |
| `export [-f csv\|json] [-r runs] [-c category] [-m model]` | Export results to CSV or JSON |
| `report RUN_ID [--open/-o]` | Generate HTML report for a run |
| `radar RUN_ID [--save/-s PATH]` | Radar/spider chart for a run |
| `leaderboard [--limit/-l N]` | Show the public leaderboard (default 20) |
| `checkpoints [--cleanup]` | List or clean up benchmark checkpoints |
| `clear-checkpoint RUN_ID` | Discard partial progress for a run |
| `download MODEL_ID [-f filename] [-d dir]` | Download a GGUF model from HuggingFace |
| `models list/add/info/search/remove/recommend` | Manage the local GGUF model catalog (`--scan` to include disk scan) |
| `recommend MODEL_PATH [--model-size/-s B]` | Optimal config recommendations for a model |
| `list-plugins` / `plugin install PACKAGE` / `plugin uninstall NAME` / `plugin info NAME` | Benchmark plugin management |
| `profile` | Detected hardware and recommended settings |
| `hw-profile` | Hardware profile and cache status |
| `verify` | Verify installation and system readiness |
| `init [--wizard/-w]` | Create config (non-interactive) or run setup wizard |
| `config --show/-s` / `config --set/-S key=value` | Inspect or update configuration |
| `validate-config` | Validate the current config file |
| `fix-config` | Auto-fix common configuration issues |
| `show-paths` | Show all FairBench directories and file locations |

## Tools

### Download models

Download GGUF models directly from HuggingFace:

```bash
# Auto-select first available GGUF file
fairbench download Qwen/Qwen2.5-1.5B-Instruct

# Download specific variant
fairbench download Qwen/Qwen2.5-1.5B-Instruct \
  --filename Qwen2.5-1.5B-Instruct-Q4_K_M.gguf

# Custom destination
fairbench download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ~/my-models/
```

Models are saved to `~/models/` by default. Download progress is shown with a terminal progress bar (percentage, speed, ETA).

### Verify system

Check FairBench installation and system readiness:

```bash
fairbench verify
```

Validates Python version, dependencies (lm-eval, llama-cpp-python), disk space, and config.

### Quick start demo

Show a demo plan with available benchmarks and models:

```bash
fairbench quick-start
```

### Setup and paths

```bash
fairbench init                # Create default config (non-interactive, scriptable)
fairbench init --wizard       # Interactive setup wizard (-w)
fairbench show-paths          # Show config & results locations
```

`fairbench init` creates `~/.fairbench/config.yaml` with sensible defaults if it doesn't already exist — non-interactive by design so it works in scripts, Dockerfiles, and CI.

> **Note:** `fairbench init --wizard` (or `-w`) runs the interactive setup wizard instead: it creates the full directory structure, prompts for leaderboard username and publishing preference (default: enabled), checks available dependencies (lm-eval, llama-cpp-python, swebench), and suggests hardware-aware settings.

`fairbench show-paths` lists all paths used by FairBench (config, results, package location) without hardcoded strings.

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
lite_mode: auto
swebench_instances: 0
swebench_split: full
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
| `lite_mode` | `auto` | Lite mode: `auto`, `enabled`, or `disabled` |
| `swebench_instances` | `0` | SWE-bench instance limit (0 = unlimited) |
| `swebench_split` | `full` | Default SWE-bench split (`full`, `lite`, `verified`) |

These are the only accepted keys. `fairbench config --set key=value` validates keys and coerces types (`publish` accepts booleans; numeric keys accept int/float). Use `fairbench validate-config` to check a file and `fairbench fix-config` to auto-repair common issues (e.g. empty values for `default_n_gpu_layers` / `default_quant`).

## Publishing

By default, every benchmark run is **automatically published** to the public leaderboard:

- Global toggle: `publish: true|false` in `~/.fairbench/config.yaml`
- Per-run override: `fairbench run MODEL --no-publish` (skip) or `--publish` (force, even if config disables it)

Publishing POSTs the report to `{api_url}/report`; if the API is unreachable the run completes normally with a warning and results stay local. See [API.md](API.md) for the payload schema.

## Docker Support

Run FairBench in a reproducible container for consistent benchmark results across environments.

### Quick Docker usage

```bash
# Build and run (auto-detects GPU support)
docker build -t fairbench .
docker run --rm -v ~/models:/app/models:ro fairbench verify

# Benchmark with GPU access (requires nvidia-container-toolkit)
docker run --rm --gpus all \
  -v ~/models:/app/models:ro \
  -v ~/.fairbench:/root/.fairbench \
  fairbench run /app/models/model.gguf --category all --lite

# Or let the CLI drive Docker (builds/updates fairbench:latest, layer-cached)
fairbench run --docker ~/models/model.gguf
fairbench run --docker --docker-image my-fairbench:1.0 ~/models/model.gguf
```

`--docker` verifies Docker availability, warns on macOS about CPU-only mode, builds the image (Docker layer cache makes repeat runs fast), detects GPU support (`nvidia` / `rocm` / `none`), and mounts model paths read-only into the container. Add `--ci` for headless runs without progress bars.

### Docker Compose (GPU support)

```yaml
version: '3.8'
services:
  fairbench:
    build: .
    volumes:
      - ./models:/app/models:ro
      - ~/.fairbench:/root/.fairbench
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with `docker compose up --build`.

### Docker considerations

| Platform | GPU support | Notes |
|----------|-------------|-------|
| Linux + NVIDIA | ✅ Full | Requires `nvidia-container-toolkit` |
| Linux + AMD | ✅ ROCm | Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| macOS (Docker Desktop) | ❌ CPU only | No GPU passthrough on macOS |
| WSL2 + Windows | ⚠️ Limited | Depends on host GPU driver |

### CI/CD with GitHub Actions

The repository ships four workflows under `.github/workflows/`:

- `ci.yml` — lint (ruff), type check (mypy), tests with coverage gate
- `release.yml` — auto version bump + PyPI publish on push to main
- `benchmark.yml` — manual benchmark run (Actions → Run workflow)
- `discover-models.yml` — weekly HuggingFace model discovery, proposes registry additions via PR

## Optional Dependencies

The published package metadata defines these extras:

| Extra | Contents | Needed for |
|-------|----------|------------|
| *(core)* | click, pyyaml, requests, psutil, rich | CLI, results, config, leaderboard |
| `[ml]` | `lm-eval>=0.4,<1.0`, `swebench>=2.0,<3.0` | Quality benchmarks, SWE-bench evaluation |
| `[all]` | `fairbench-ai[ml]` | Convenience alias for the ML stack |
| `[dev]` | pytest, pytest-cov, ruff, mypy, huggingface-hub | Contributing / CI |

`llama-cpp-python` is **not** part of any extra (it requires C build tools). Install it separately for speed/memory benchmarks:

```bash
pip install llama-cpp-python          # tries precompiled wheels first
sudo apt-get install cmake build-essential   # if a source build is needed (Debian/Ubuntu)
brew install cmake llvm                      # macOS equivalent
```

When installing from a source checkout (`python setup.py install`), the custom `setup.py` attempts this automatically with wheel preference (`LLAMA_CPP_FORCE_BUILD=0`) and skips it in CI environments.

## Requirements

- Python 3.10+
- [llama-cpp-python](https://github.com/ggerganov/llama.cpp) for speed/memory benchmarks
