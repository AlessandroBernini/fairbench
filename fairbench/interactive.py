"""Interactive mode for FairBench — guided benchmark configuration."""

from __future__ import annotations

import os

import click

from fairbench.config import load_config
from fairbench.engine import BenchmarkEngine
from fairbench.models import ModelCatalog
from fairbench.types import BenchmarkCategory


def _select_model(catalog: ModelCatalog) -> str:
    """Guide user through model selection."""
    models = catalog.list_models()

    if models:
        click.echo("\n📚 Available Models:")
        click.echo("-" * 80)
        click.echo(f"  {'#':<4s} {'Name':<35s} {'Arch':<12s} {'Size (GB)':>10s} {'Quant':<10s}")
        click.echo("-" * 80)
        for idx, m in enumerate(models, 1):
            click.echo(
                f"  {idx:<4d} {m.name:<35s} {m.architecture:<12s} {m.file_size_gb:>10.2f} {m.quantization:<10s}"
            )
        click.echo()

        choice = click.prompt(
            "Select model number",
            type=int,
            default=1,
        )
        if 1 <= choice <= len(models):
            return str(models[choice - 1].path)

    click.echo("\n⚠️  No catalog models found or invalid selection.")
    path = click.prompt(
        "Enter model path or catalog name",
        type=str,
    )
    return path


def _select_category() -> str | None:
    """Guide user through benchmark category selection."""
    categories = ["quality", "speed", "memory", "swebench", "all"]
    click.echo("\n⚙️  Benchmark Categories:")
    for idx, cat in enumerate(categories, 1):
        click.echo(f"  {idx}. {cat}")

    choice = click.prompt(
        "Select category",
        type=int,
        default=1,
    )
    if 1 <= choice <= len(categories):
        return categories[choice - 1]
    return "all"


def _configure_parameters(
    model_path: str,
) -> dict[str, object]:
    """Guide user through parameter configuration."""
    params: dict[str, object] = {}

    # n_ctx
    n_ctx = click.prompt(
        "Context window size (n_ctx)",
        type=int,
        default=4096,
    )
    params["n_ctx"] = n_ctx

    # n_gpu_layers
    n_gpu = click.prompt(
        "GPU layers to offload (n_gpu_layers)",
        type=int,
        default=-1,
    )
    params["n_gpu_layers"] = n_gpu

    # n_threads
    n_threads = click.prompt(
        "CPU threads",
        type=int,
        default=os.cpu_count() or 4,
    )
    params["n_threads"] = n_threads

    # quantization
    quant = click.prompt(
        "Quantization type (leave empty for default)",
        type=str,
        default="",
    )
    params["quantization"] = quant if quant else None

    return params


def _review_and_confirm(
    model_path: str,
    category: str,
    params: dict[str, object],
) -> bool:
    """Show summary and ask for confirmation."""
    click.echo("\n" + "=" * 60)
    click.echo("📋 Configuration Summary")
    click.echo("=" * 60)
    click.echo(f"  Model:      {model_path}")
    click.echo(f"  Category:   {category}")
    click.echo(f"  n_ctx:      {params.get('n_ctx')}")
    click.echo(f"  n_gpu_layers: {params.get('n_gpu_layers')}")
    click.echo(f"  n_threads:  {params.get('n_threads')}")
    quant = params.get("quantization")
    click.echo(f"  quantization: {quant if quant else 'default'}")
    click.echo("=" * 60)

    return click.confirm("Run benchmark with these settings?", default=True)


def _run_interactive() -> None:
    """Execute the interactive benchmark flow."""
    click.echo("\n🚀 FairBench Interactive Mode")
    click.echo("Follow the prompts to configure your benchmark.\n")

    # Step 1: Model selection
    catalog = ModelCatalog()
    model_path = _select_model(catalog)

    # Step 2: Category selection
    category = _select_category()

    # Step 3: Parameter configuration
    params = _configure_parameters(model_path)

    # Step 4: Review & confirm
    if not _review_and_confirm(model_path, category, params):
        click.echo("\n❌ Benchmark cancelled.")
        return

    # Step 5: Execute
    click.echo("\n⏳ Running benchmark...")
    config = load_config()
    engine = BenchmarkEngine(config)

    cat_map = {
        "quality": BenchmarkCategory.QUALITY,
        "speed": BenchmarkCategory.SPEED,
        "memory": BenchmarkCategory.MEMORY,
        "swebench": BenchmarkCategory.SWE_BENCH,
        "all": None,
    }
    cat = cat_map.get(category)

    if category == "all":
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.QUALITY,
            n_ctx=params.get("n_ctx"),
            n_gpu_layers=params.get("n_gpu_layers"),
            n_threads=params.get("n_threads"),
            quantization=params.get("quantization"),
        )
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.SPEED,
            n_ctx=params.get("n_ctx"),
            n_gpu_layers=params.get("n_gpu_layers"),
            n_threads=params.get("n_threads"),
            quantization=params.get("quantization"),
        )
        engine.run_benchmark(
            model_path=model_path,
            category=BenchmarkCategory.MEMORY,
            n_ctx=params.get("n_ctx"),
            n_gpu_layers=params.get("n_gpu_layers"),
            n_threads=params.get("n_threads"),
            quantization=params.get("quantization"),
        )
    elif cat:
        engine.run_benchmark(
            model_path=model_path,
            category=cat,
            n_ctx=params.get("n_ctx"),
            n_gpu_layers=params.get("n_gpu_layers"),
            n_threads=params.get("n_threads"),
            quantization=params.get("quantization"),
        )
    else:
        click.echo("❌ Invalid category selected.")
        return

    click.echo("\n✅ Benchmark complete!")
    click.echo("   View results: fairbench results")
    click.echo("   Generate report: fairbench report <run_id>")


@click.command("interactive")
def interactive_cmd() -> None:
    """Guided interactive benchmark configuration and execution."""
    try:
        _run_interactive()
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Interrupted by user.")
    except RuntimeError as exc:
        click.echo(f"\n❌ Error: {exc}")
