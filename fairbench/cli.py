"""CLI for FairBench."""

from __future__ import annotations

import click

from fairbench.config import load_config, save_config
from fairbench.engine import BenchmarkEngine
from fairbench.swebench import SWEBENCH_SOURCES
from fairbench.types import BenchmarkCategory


@click.group()
@click.version_option(version="0.3.0")
@click.option("--config", "-c", "config_path", help="Path to config file", default=None)
@click.pass_context
def cli(ctx, config_path):
    """FairBench - AI model benchmarking suite with SWE-bench integration.

    FairBench evaluates AI models across multiple dimensions:
    - Quality: MMLU, HellaSwag, GSM8K, HumanEval, IFEval, TruthfulQA, ARC-Challenge, LiveBench
    - Speed: TTFT, TPOT, throughput
    - Memory: VRAM and RAM peak usage
    - SWE-bench: GitHub issue resolution capability
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@cli.command()
@click.argument("model_path")
@click.option(
    "--category",
    "-k",
    "category",
    type=click.Choice(["quality", "speed", "memory", "swebench", "all"]),
    default="all",
    help="Benchmark category to run",
)
@click.option("--tasks", "-t", "tasks", multiple=True, help="Specific quality benchmarks to run")
@click.option("--n-ctx", "-n", default=None, type=int, help="Context window size")
@click.option("--n-gpu-layers", "-g", default=None, type=int, help="GPU layers to offload")
@click.option("--n-threads", "-j", default=None, type=int, help="CPU threads")
@click.option("--quant", "-q", default=None, help="Quantization type")
@click.option("--lite/--no-lite", default=None, help="Use lite mode for quality benchmarks")
@click.option(
    "--swebench-split",
    "-s",
    "swebench_split",
    type=click.Choice(list(SWEBENCH_SOURCES.keys())),
    default=None,
    help="SWE-bench split to use",
)
@click.option(
    "--swebench-limit",
    "-l",
    "swebench_limit",
    type=int,
    default=None,
    help="Limit SWE-bench instances",
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
    help="SWE-bench split",
)
@click.option("--limit", "-l", type=int, default=10, help="Number of instances to show")
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
    help="SWE-bench split",
)
@click.option("--limit", "-l", type=int, default=5, help="Number of instances to evaluate")
@click.option("--repo-filter", "-r", "repo_filter", help="Filter by repository")
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
@click.option("--show", "-s", "show", is_flag=True, help="Show current config")
@click.option("--set", "-S", "set_kv", multiple=True, help="Set config key=value")
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
@click.option("--limit", "-l", type=int, default=20, help="Number of results to show")
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
@click.option("--category", "-c", help="Filter by category")
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


if __name__ == "__main__":
    cli()
