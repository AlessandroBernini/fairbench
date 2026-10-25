"""CLI for FairBench."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from fairbench.config import Config
from fairbench.registry import get_model_info, list_models
from fairbench.backend import detect_hardware
from fairbench.reporter import Reporter
from fairbench.engine import run_benchmark
from fairbench.quality import get_available_tasks, get_task_info, DEFAULT_TASKS

console = Console()


@click.group()
def cli():
    """FairBench - Fair benchmarking for local LLMs under 70B on consumer hardware."""
    pass


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path to GGUF file")
@click.option("--quant", "-q", default=None, help="Quantization (default: from config)")
@click.option("--n-gpu-layers", "-ngl", default=None, help="Number of GPU layers (default: from config)")
@click.option("--n-threads", "-t", default=None, help="Number of CPU threads (default: auto)")
@click.option("--n-tokens", "-n", default=512, help="Number of tokens to generate (default: 512)")
@click.option("--lite/--full", default=True, help="Use lite mode for quality benchmark (default: lite)")
@click.option("--tasks", "-t", default=None, help=f"Benchmarks to run (comma-separated). Default: {','.join(DEFAULT_TASKS)}. Available: {', '.join(get_available_tasks())}")
@click.option("--publish/--no-publish", "-p/-np", default=None, help="Publish to leaderboard (default: from config)")
def run(model: str, quant: str, n_gpu_layers: int, n_threads: int, n_tokens: int, lite: bool, tasks: str, publish):
    """Run a benchmark on the specified model."""
    config = Config()

    # Parse tasks from CLI
    if tasks:
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = None  # Will default to DEFAULT_TASKS in engine

    # Determine publish setting
    if publish is None:
        publish = config.publish_default

    # Resolve model path
    model_path = Path(model)
    if not model_path.exists():
        # Try to get from registry
        model_info = get_model_info(model)
        if model_info.get('repo'):
            console.print(f"[yellow]Model '{model}' not found locally. Please download the GGUF first.[/yellow]")
            console.print(f"  Source: {model_info['repo']}")
            console.print(f"  File pattern: {model_info['file_pattern']}")
            console.print(f"\n[bold]Download with:[/bold]")
            console.print(f"  huggingface-cli download {model_info['repo']} {model_info['file_pattern']} --local-dir ~/.fairbench/models/")
            return
        else:
            console.print(f"[red]Error: '{model}' not found and not in registry.[/red]")
            return

    # Determine GPU layers
    if n_gpu_layers is None:
        n_gpu_layers = config.default_n_gpu_layers

    # Determine threads
    if n_threads is None:
        n_threads = config.default_threads

    # Detect hardware
    hardware = detect_hardware()
    console.print("[bold green]Hardware detected:[/bold green]")
    console.print(f"  CPU: {hardware['cpu']['model']} ({hardware['cpu']['cores']} cores / {hardware['cpu']['threads']} threads)")
    console.print(f"  RAM: {hardware['ram_gb']} GB")
    if hardware['gpu']['type'] != 'none':
        console.print(f"  GPU: {hardware['gpu']['model']} ({hardware['gpu']['vram_gb']} GB VRAM)")
    else:
        console.print("  GPU: None (CPU-only mode)")
    console.print(f"  OS: {hardware['os']}")

    # Get model info
    model_info = get_model_info(model)
    if model_info['repo']:
        console.print(f"\n[bold blue]Model:[/bold blue] {model} ({model_info['size_b']}B params)")
        console.print(f"  Source: {model_info['repo']}")
    else:
        console.print(f"\n[bold blue]Model:[/bold blue] {model} (custom GGUF)")

    # Run benchmark via engine
    console.print(f"\n[bold]Config:[/bold]")
    console.print(f"  Quantization: {quant or config.default_quant}")
    console.print(f"  GPU layers: {n_gpu_layers}")
    console.print(f"  Threads: {n_threads}")
    console.print(f"  Tokens: {n_tokens}")
    console.print(f"  Lite mode: {'Yes' if lite else 'No'}")
    console.print(f"  Publish: {'Yes' if publish else 'No'}")
    if task_list:
        console.print(f"  Tasks: {', '.join(task_list)}")
    else:
        console.print(f"  Tasks: {', '.join(DEFAULT_TASKS)} (default)")

    report_path = run_benchmark(
        model_path=model_path,
        model_name=model,
        quantization=quant or config.default_quant,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_tokens=n_tokens,
        lite=lite,
        publish=publish,
        tasks=task_list,
    )
    console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")


@cli.command()
@click.option("--latest", "-l", is_flag=True, help="Show latest report")
def report(latest: bool):
    """View benchmark reports."""
    config = Config()
    reporter = Reporter(config.results_dir)

    if latest:
        report = reporter.get_latest()
        if report:
            _print_report(report)
        else:
            console.print("[yellow]No reports found.[/yellow]")
    else:
        reports = reporter.list_reports()
        if reports:
            table = Table(title="FairBench Reports")
            table.add_column("ID")
            table.add_column("Timestamp")
            table.add_column("Model")
            table.add_column("Published")
            for r in reports:
                table.add_row(
                    r["id"],
                    r["timestamp"],
                    r["model"],
                    "Yes" if r["published"] else "No"
                )
            console.print(table)
        else:
            console.print("[yellow]No reports found.[/yellow]")


@cli.command()
def publish():
    """Publish latest report to leaderboard."""
    config = Config()
    reporter = Reporter(config.results_dir)

    report = reporter.get_latest()
    if not report:
        console.print("[yellow]No reports to publish.[/yellow]")
        return

    report["published"] = True
    # In production, this would POST to the Vercel API
    # For now, just update the local file
    filepath = config.results_dir / f"{report['timestamp']}_{report['model']['name']}_{report['id']}.json"
    import json
    filepath.write_text(json.dumps(report, indent=2))

    console.print(f"[bold green]Report {report['id']} marked as published[/bold green]")


@cli.command()
def list_benchmarks():
    """List all available benchmark categories."""
    console.print("[bold]Available benchmarks:[/bold]")
    console.print(f"  {'Name':<15} {'Type':<20} {'Lite Mode'}")
    console.print(f"  {'-'*15} {'-'*20} {'-'*30}")
    for task in get_available_tasks():
        info = get_task_info(task)
        console.print(
            f"  {info['name']:<15} {info['type']:<20} {info['lite']}"
        )
    console.print(f"\n[bold]Usage:[/bold]")
    console.print(f"  fairbench run --model <model> --tasks mmlu,humaneval")
    console.print(f"  Default tasks: {', '.join(DEFAULT_TASKS)}")


@cli.command()
def list_models():
    """List all available models in the registry."""
    models = list_models()
    console.print("[bold]Available models:[/bold]")
    for name in models:
        info = get_model_info(name)
        if info['repo']:
            console.print(f"  {name} ({info['size_b']}B) - {info['repo']}")
        else:
            console.print(f"  {name} (custom)")


def _print_report(report: dict):
    """Print a formatted report."""
    console.print(f"\n[bold]Report: {report['id']}[/bold]")
    console.print(f"Timestamp: {report['timestamp']}")
    console.print(f"Model: {report['model']['name']} ({report['model']['quantization']})")

    console.print("\n[bold]Hardware:[/bold]")
    hw = report['hardware']
    console.print(f"  CPU: {hw['cpu']['model']}")
    console.print(f"  RAM: {hw['ram_gb']} GB")
    if hw['gpu']['type'] != 'none':
        console.print(f"  GPU: {hw['gpu']['model']} ({hw['gpu']['vram_gb']} GB)")

    console.print("\n[bold]Quality:[/bold]")
    for k, v in report['quality'].items():
        console.print(f"  {k}: {v:.3f}")

    console.print("\n[bold]Speed:[/bold]")
    for k, v in report['speed'].items():
        console.print(f"  {k}: {v:.2f}")

    console.print("\n[bold]Memory:[/bold]")
    for k, v in report['memory'].items():
        console.print(f"  {k}: {v:.2f} GB")


if __name__ == "__main__":
    cli()
