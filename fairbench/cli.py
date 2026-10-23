"""CLI for FairBench."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from fairbench.config import Config
from fairbench.registry import get_model_info, list_models
from fairbench.backend import detect_hardware
from fairbench.reporter import Reporter

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
@click.option("--publish/--no-publish", "-p/-np", default=None, help="Publish to leaderboard (default: from config)")
def run(model: str, quant: str, n_gpu_layers: int, n_threads: int, n_tokens: int, publish: Optional[bool]):
    """Run a benchmark on the specified model."""
    config = Config()

    # Determine publish setting
    if publish is None:
        publish = config.publish_default

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

    # Run benchmarks (placeholder - will be implemented)
    console.print("\n[bold yellow]Running benchmarks...[/bold yellow]")
    console.print("  Quality (MMLU, HellaSwag, GSM8K)...")
    console.print("  Speed (TTFT, TPOT, Throughput)...")
    console.print("  Memory (VRAM, CPU RAM)...")

    # Generate report
    reporter = Reporter(config.results_dir)
    quality = {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}  # Placeholder
    speed = {"ttft_ms": 0.0, "tpot_ms": 0.0, "throughput": 0.0}  # Placeholder
    memory = {"vram_peak_gb": 0.0, "cpu_ram_peak_gb": 0.0}  # Placeholder

    report_path = reporter.generate_report(
        model_name=model,
        quantization=quant or config.default_quant,
        hardware=hardware,
        quality=quality,
        speed=speed,
        memory=memory,
        publish=publish,
    )

    console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")
    if publish:
        console.print(f"[bold green]Report published to leaderboard[/bold green]")
    else:
        console.print(f"[bold yellow]Report saved locally (not published)[/bold yellow]")


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
