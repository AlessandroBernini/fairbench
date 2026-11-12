"""CLI for FairBench."""

from __future__ import annotations

import os
import re
from importlib import metadata

import click

from fairbench.config import load_config, save_config
from fairbench.engine import BenchmarkEngine
from fairbench.swebench import SWEBENCH_SOURCES
from fairbench.types import BenchmarkCategory

# Resolve version at module load time (handles CI where package isn't installed)
try:
    _cli_version = metadata.version("fairbench-ai")
except metadata.PackageNotFoundError:
    _cli_version = "0.0.0.dev"  # dev/CI fallback


@click.group()
@click.version_option(version=_cli_version, prog_name="FairBench")
@click.option("--config", "-c", "config_path", help="Path to config file", default=None)
@click.pass_context
def cli(ctx, config_path):
    """FairBench — AI model benchmarking suite with SWE-bench integration.

    Evaluate your GGUF models across quality, speed, memory, and coding benchmarks.

    Quick Start:
      fairbench init          Interactive setup wizard
      fairbench download      Download a GGUF model from HuggingFace
      fairbench verify        Check system readiness
      fairbench run MODEL     Benchmark a GGUF model
      fairbench show-paths    Show config & results locations
      fairbench quick-start   Demo with available benchmarks
      fairbench list-benchmarks  List available quality benchmarks

    Examples:
      fairbench init
      fairbench download Qwen/Qwen2.5-1.5B-Instruct
      fairbench run ~/models/qwen2.5-7b.gguf
      fairbench run model.gguf --category quality --tasks mmlu hellaswag
      fairbench run model.gguf --category speed
      fairbench verify
      fairbench quick-start
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@cli.command()
def init():
    """Interactive setup wizard — creates config, checks dependencies, shows paths."""
    from fairbench.install import cmd_init
    cmd_init()


@click.command("download")
@click.argument("model_id", required=True)
@click.option("--filename", "-f", help="Specific GGUF file to download")
@click.option("--local-dir", "-d", default=None, help="Target directory (default: ~/models/)")
def download(model_id, filename, local_dir):
    """Download a GGUF model from HuggingFace.
    
    MODEL_ID is the HuggingFace repository ID (e.g., 'Qwen/Qwen2.5-1.5B-Instruct').
    
    Examples:
    \b
    fairbench download Qwen/Qwen2.5-1.5B-Instruct
    fairbench download Qwen/Qwen2.5-1.5B-Instruct --filename Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    """
    from fairbench.install import cmd_download
    cmd_download(model_id=model_id, filename=filename, local_dir=local_dir)


@cli.command("show-paths")
def show_paths():
    """Show all FairBench directories and file locations."""
    from fairbench.install import cmd_show_paths
    cmd_show_paths()


@cli.command()
def verify():
    """Verify FairBench installation and system readiness.
    
    Checks Python version, config files, dependencies, and disk space.
    Shows actionable guidance for any issues found.
    """
    from fairbench.install import cmd_verify
    cmd_verify()


@cli.command("quick-start")
def quick_start():
    """Show a demo plan with available benchmarks and models."""
    from fairbench.install import cmd_quick_start
    cmd_quick_start()


@cli.command()
@click.argument("model_path")
@click.option(
    "--category",
    "-k",
    "category",
    type=click.Choice(["quality", "speed", "memory", "swebench", "all"]),
    default="all",
    help="Benchmark category to run (-k shorthand)",
)
@click.option("--tasks", "-t", "tasks", multiple=True, help="Specific quality benchmarks to run (-t shorthand)")
@click.option("--n-ctx", "-n", default=None, type=int, help="Context window size (-n shorthand)")
@click.option("--n-gpu-layers", "-g", default=None, type=int, help="GPU layers to offload (-g shorthand)")
@click.option("--n-threads", "-j", default=None, type=int, help="CPU threads (-j shorthand)")
@click.option("--quant", "-q", default=None, help="Quantization type (-q shorthand)")
@click.option("--lite/--no-lite", default=None, help="Use lite mode for quality benchmarks")
@click.option(
    "--swebench-split",
    "-s",
    "swebench_split",
    type=click.Choice(list(SWEBENCH_SOURCES.keys())),
    default=None,
    help="SWE-bench split to use (-s shorthand)",
)
@click.option(
    "--swebench-limit",
    "-l",
    "swebench_limit",
    type=int,
    default=None,
    help="Limit SWE-bench instances (-l shorthand)",
)
@click.option("--publish/--no-publish", default=None, help="Override publish setting")
@click.pass_context
def run(
    ctx,
    model_path,
    category,
    tasks,
    n_ctx,
    n_gpu_layers,
    n_threads,
    quant,
    lite,
    swebench_split,
    swebench_limit,
    publish,
):
    """Run benchmarks on a model.

    MODEL_PATH is the path to the GGUF model file.

    Examples:
    \b
    fairbench run ~/models/qwen3-35b-q4_k_m.gguf
    fairbench run ~/models/model.gguf --category quality --tasks mmlu hellaswag
    fairbench run ~/models/model.gguf --category swebench --swebench-split lite
    fairbench run ~/models/model.gguf --category all --lite
    """
    config = ctx.obj["config"]

    # Override publish setting if specified
    if publish is not None:
        config.publish = publish

    engine = BenchmarkEngine(config)

    # Map category string to enum
    category_map = {
        "quality": BenchmarkCategory.QUALITY,
        "speed": BenchmarkCategory.SPEED,
        "memory": BenchmarkCategory.MEMORY,
        "swebench": BenchmarkCategory.SWE_BENCH,
        "all": None,  # Will run all
    }

    cat = category_map.get(category)

    # Handle SWE-bench special cases
    if category == "swebench" or (category == "all" and swebench_split):
        lite_mode = "enabled" if lite else "disabled"
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.SWE_BENCH,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            quantization=quant,
            lite_mode=lite_mode,
        )
        # Override split if specified
        if swebench_split:
            engine.swebench_client.config = config
            engine.swebench_client.config.swebench_split = swebench_split
        return

    if category == "all":
        # Run all benchmarks
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.QUALITY,
            tasks=list(tasks) if tasks else None,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            quantization=quant,
            lite_mode="enabled" if lite else "disabled",
        )
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.SPEED,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            quantization=quant,
        )
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.MEMORY,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            quantization=quant,
        )
        return

    if cat is None:
        click.echo("Invalid category")
        return

    engine.run_benchmark(
        model_path=model_path,
        category=cat,
        tasks=list(tasks) if tasks else None,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        quantization=quant,
        lite_mode="enabled" if lite else "disabled",
    )


@cli.command("list-benchmarks")
@click.pass_context
def list_benchmarks(ctx):
    """List available quality benchmarks."""
    engine = BenchmarkEngine(ctx.obj["config"])
    benchmarks = engine.list_available_benchmarks()

    click.echo("\nAvailable Quality Benchmarks:")
    click.echo("-" * 50)
    for name, info in benchmarks.items():
        lite_marker = " [lite]" if info.get("lite") else ""
        click.echo(f"  {name:20s} {info['name']:25s}{lite_marker}")
        click.echo(f"    {info['description']}")
    click.echo()


@cli.command("list-swebench")
@click.option(
    "--split",
    "-s",
    "split",
    type=click.Choice(list(SWEBENCH_SOURCES.keys())),
    default="full",
    help="SWE-bench split (-s shorthand)",
)
@click.option("--limit", "-l", type=int, default=10, help="Number of instances to show (-l shorthand)")
@click.pass_context
def list_swebench(ctx, split, limit):
    """List available SWE-bench instances."""
    engine = BenchmarkEngine(ctx.obj["config"])
    instances = engine.get_swebench_instances(split=split, limit=limit)

    click.echo(f"\nSWE-bench '{split}' instances (showing {len(instances)}):")
    click.echo("-" * 70)
    click.echo(f"  {'Instance ID':<35s} {'Repo':<25s} {'Language':<10s}")
    click.echo("-" * 70)
    for inst in instances:
        click.echo(
            f"  {inst['instance_id']:<35s} {inst['repo']:<25s} {inst['code_language']:<10s}"
        )
    click.echo()


@cli.command("swebench-eval")
@click.argument("model_path")
@click.option(
    "--split",
    "-s",
    "split",
    type=click.Choice(list(SWEBENCH_SOURCES.keys())),
    default="full",
    help="SWE-bench split (-s shorthand)",
)
@click.option("--limit", "-l", type=int, default=5, help="Number of instances to evaluate (-l shorthand)")
@click.option("--repo-filter", "-r", "repo_filter", help="Filter by repository (-r shorthand)")
@click.pass_context
def swebench_eval(ctx, model_path, split, limit, repo_filter):
    """Run SWE-bench evaluation on a model.

    MODEL_PATH is the path to the GGUF model file.
    """
    engine = BenchmarkEngine(ctx.obj["config"])

    click.echo("\nSWE-bench Evaluation")
    click.echo(f"Model: {model_path}")
    click.echo(f"Split: {split}")
    click.echo(f"Limit: {limit}")
    if repo_filter:
        click.echo(f"Repo filter: {repo_filter}")

    # Fetch instances
    instances = engine.swebench_client.fetch_instances(
        split=split,
        limit=limit,
        repo_filter=repo_filter,
    )

    if not instances:
        click.echo("No instances available")
        return

    click.echo(f"\nFound {len(instances)} instances")

    # Run evaluation
    results = engine.swebench_client.run_evaluation(
        model_path=model_path,
        instances=instances,
        split=split,
        limit=limit,
    )

    # Show results
    stats = engine.swebench_client.compute_stats(results)
    click.echo(f"\n{'=' * 50}")
    click.echo("Results:")
    click.echo(f"  Total: {stats.total_instances}")
    click.echo(f"  Resolved: {stats.resolved}")
    click.echo(f"  Failed: {stats.failed}")
    click.echo(f"  Errors: {stats.errors}")
    click.echo(f"  Score: {stats.avg_test_score:.1f}%")
    click.echo(f"{'=' * 50}")


@cli.command("config")
@click.option("--show", "-s", "show", is_flag=True, help="Show current config (-s shorthand)")
@click.option("--set", "-S", "set_kv", multiple=True, help="Set config key=value (-S shorthand)")
@click.pass_context
def config_cmd(ctx, show, set_kv):
    """Manage FairBench configuration."""
    config = ctx.obj["config"]

    if show:
        click.echo("\nCurrent Configuration:")
        click.echo("-" * 40)
        for key, value in config.__dict__.items():
            click.echo(f"  {key}: {value}")
        click.echo()
        return

    if set_kv:
        for kv in set_kv:
            if "=" not in kv:
                click.echo(f"Invalid key=value: {kv}")
                continue
            key, value = kv.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Parse value types
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string

            if hasattr(config, key):
                setattr(config, key, value)
                click.echo(f"Set {key} = {value}")
            else:
                click.echo(f"Unknown config key: {key}")

        # Save config
        save_config(config)
        click.echo("\nConfig saved")


@cli.command("leaderboard")
@click.option("--limit", "-l", type=int, default=20, help="Number of results to show (-l shorthand)")
@click.pass_context
def leaderboard(ctx, limit):
    """Show the leaderboard."""
    engine = BenchmarkEngine(ctx.obj["config"])
    results = engine.reporter.get_leaderboard(limit=limit)

    if not results:
        click.echo("No results available")
        return

    click.echo("\nLeaderboard:")
    click.echo("-" * 80)
    click.echo(f"  {'Model':<30s} {'Quality':<10s} {'Speed':<10s} {'Memory':<10s} {'SWE':<10s}")
    click.echo("-" * 80)

    for result in results:
        model = result.get("model_name", "unknown")[:30]
        quality = result.get("quality_scores", {})
        avg_quality = sum(quality.values()) / len(quality) if quality else 0
        speed = result.get("speed", {}).get("throughput_tokens_per_sec", 0)
        memory = result.get("memory", {}).get("vram_peak_mb", 0)
        swebench = result.get("swebench", {})
        sweb_score = (
            (swebench.get("resolved", 0) / swebench.get("total", 1) * 100)
            if swebench.get("total", 0) > 0
            else 0
        )

        click.echo(
            f"  {model:<30s} {avg_quality:<10.2f} {speed:<10.1f} {memory:<10.0f} {sweb_score:<10.1f}"
        )
    click.echo()


@cli.command("results")
@click.option("--category", "-c", help="Filter by category (-c shorthand)")
@click.pass_context
def results_cmd(ctx, category):
    """List local benchmark results."""
    data = BenchmarkEngine(ctx.obj["config"]).data
    results = data.list_results()

    if not results:
        click.echo("No local results")
        return

    click.echo(f"\nLocal results ({len(results)}):")
    for r in results:
        click.echo(f"  {r}")


@cli.command("init")
@click.pass_context
def init_cmd(ctx):
    """Initialize FairBench configuration.

    Creates ~/.fairbench/config.yaml with sensible defaults if it doesn't already exist.
    """
    from fairbench.config import CONFIG_PATH, _default_config, save_config

    if CONFIG_PATH.exists():
        click.echo(f"⚠️  Config already exists at {CONFIG_PATH}")
        click.echo("   Delete it first or use 'fairbench config --show' to inspect.")
        return

    config = _default_config()
    save_config(config)
    click.echo(f"✅ Config created at {CONFIG_PATH}")
    click.echo("   Edit it with: fairbench config --set key=value")


@cli.command("profile")
def profile_cmd():
    """Show detected hardware and recommended settings."""
    from fairbench.hardware import detect_hardware, generate_adaptive_config

    hw = detect_hardware()
    adaptive = generate_adaptive_config(hw)

    click.echo("\n🔍 Hardware Profile")
    click.echo("=" * 50)
    click.echo(f"  OS:           {hw.os_platform}")
    click.echo(f"  CPU:          {hw.cpu_model}")
    click.echo(f"  Cores:        {hw.cpu_cores}")
    click.echo(f"  Threads:      {hw.cpu_threads}")
    click.echo(f"  RAM:          {hw.total_ram_gb:.1f} GB")
    click.echo(f"  GPU:          {'Yes' if hw.gpu_available else 'No'}")
    if hw.gpu_available:
        click.echo(f"    Type:       {hw.gpu_type}")
        click.echo(f"    VRAM:       {hw.gpu_vram_gb:.1f} GB")
    click.echo()
    click.echo("💡 Recommended Settings")
    click.echo("-" * 50)
    click.echo(f"  n_ctx:        {adaptive['default_n_ctx']}")
    click.echo(f"  n_gpu_layers: {adaptive['default_n_gpu_layers']}")
    click.echo(f"  threads:      {adaptive['default_threads']}")
    click.echo(f"  quant:        {adaptive['default_quant']}")
    click.echo()


@cli.command("recommend")
@click.argument("model_path", required=False, default=None)
@click.option(
    "--model-size",
    "-s",
    "model_size_gb",
    type=float,
    default=None,
    help="Model size in GB (auto-detected from filename if possible)",
)
@click.pass_context
def recommend_cmd(ctx, model_path, model_size_gb):
    """Show optimal config recommendations for a model.

    MODEL_PATH is the path to the GGUF model file (optional — can also use --model-size).

    Examples:
    \\b
    fairbench recommend ~/models/qwen2.5-7b.gguf
    fairbench recommend --model-size 4.0
    """
    from fairbench.hardware import detect_hardware, generate_recommendation

    hw = detect_hardware()

    # Auto-detect model size from filename if not provided
    if model_size_gb is None and model_path:
        model_size_gb = _estimate_model_size(model_path)

    if model_size_gb is None:
        click.echo("  ℹ️  No model size provided. Use --model-size or pass a model path.")
        click.echo(f"  ℹ️  Hardware detected: {hw.total_ram_gb:.1f} GB RAM, {hw.cpu_threads} threads")
        click.echo()
        click.echo("  💡 Run 'fairbench profile' to see hardware details.")
        return

    recommendations = generate_recommendation(hw, model_size_gb)

    click.echo("\n📋 Recommendations")
    click.echo("=" * 50)
    click.echo(f"  Model size:   {model_size_gb:.1f} GB")
    click.echo(f"  Available RAM: {hw.total_ram_gb:.1f} GB")
    click.echo()

    if recommendations.get("warning"):
        click.echo(f"  ⚠️  {recommendations['warning']}")
        click.echo()

    click.echo("  Recommended settings:")
    click.echo(f"    n_gpu_layers: {recommendations['gpu_layers']}")
    click.echo(f"      → {recommendations['gpu_note']}")
    click.echo()
    click.echo(f"    quantization: {recommendations['quant']}")
    click.echo(f"      → {recommendations['quant_note']}")
    click.echo()
    click.echo(f"    n_ctx:        {recommendations['n_ctx']}")
    click.echo(f"      → {recommendations['n_ctx_note']}")
    click.echo()
    click.echo(f"    threads:      {recommendations['threads']}")
    click.echo(f"      → {recommendations['threads_note']}")
    click.echo()


def _estimate_model_size(model_path: str) -> float | None:
    """Estimate model size in GB from filename or file size."""
    # Try to get actual file size
    try:
        size_bytes = os.path.getsize(model_path)
        if size_bytes > 0:
            return round(size_bytes / (1024**3), 2)
    except (OSError, FileNotFoundError):
        pass

    # Try to parse from filename (e.g., "qwen2.5-7b" → 7 GB, "llama3-8b" → 8 GB)
    basename = os.path.basename(model_path).lower()
    # Match patterns like "7b", "8b", "13b", "70b", "1.5b", "3b"
    match = re.search(r"(\d+\.?\d*)\s*b", basename)
    if match:
        return float(match.group(1))

    return None


if __name__ == "__main__":
    cli()
