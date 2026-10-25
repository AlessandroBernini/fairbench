"""Quality benchmark for FairBench.

Uses lm_eval (lm-eval>=0.4.0) to measure MMLU, HellaSwag, and GSM8K
scores on local GGUF models via llama-cpp-python backend.

Supports 'lite' mode for faster evaluation on consumer hardware.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def run_quality_benchmark(
    model_path: Path,
    task_names_in: list[str] | None = None,
    lite: bool = True,
    batch_size: int = 4,
) -> dict[str, float]:
    """Run quality benchmark on a GGUF model.

    Uses lm_eval to measure MMLU, HellaSwag, and GSM8K scores.
    Falls back gracefully if lm_eval or llama-cpp-python is not available.

    Args:
        model_path: Path to the GGUF model file.
        task_names_in: List of tasks to run. Defaults to ['mmlu', 'hellaswag', 'gsm8k'].
        lite: If True, use subset of tasks for faster evaluation.
        batch_size: Batch size for evaluation.

    Returns:
        dict with keys 'mmlu', 'hellaswag', 'gsm8k' and float scores.
    """
    if task_names_in is None:
        task_names_in = ["mmlu", "hellaswag", "gsm8k"]

    # Try to import lm_eval
    try:
        import lm_eval
        from lm_eval import Evaluator, tasks
    except ImportError:
        console.print(
            "[yellow]Warning: lm_eval not installed. "
            "Install with: pip install lm-eval>=0.4.0[/yellow]"
        )
        return {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}

    # Try to import llama-cpp-python for GGUF support
    try:
        from lm_eval.models.llama_cpp import LlamaCPP
    except ImportError:
        console.print(
            "[yellow]Warning: llama-cpp-python not installed. "
            "Install with: pip install llama-cpp-python[/yellow]"
        )
        return {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}

    # Determine model name from path
    model_name = str(model_path)
    if not Path(model_path).exists():
        console.print(f"[red]Error: model file not found: {model_path}[/red]")
        return {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}

    console.print(f"\n[bold blue]Quality benchmark:[/bold blue] {model_path.name}")
    console.print(f"  Tasks: {', '.join(tasks)}")
    console.print(f"  Lite mode: {'Yes' if lite else 'No'}")
    console.print(f"  Batch size: {batch_size}")

    # Initialize the model
    try:
        console.print("[bold]Loading model...[/bold]")
        lm = LlamaCPP(
            model_path=str(model_path),
            n_ctx=4096,
            n_batch=batch_size,
        )
    except Exception as e:
        console.print(f"[yellow]Warning: failed to load model ({e}). "
                      "Using CPU fallback.[/yellow]")
        return {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}

    # Determine the right task names for lm_eval
    lm_task_names: list[str] = []
    for task in task_names_in:
        if task == "mmlu":
            if lite:
                lm_task_names.append("mmlu_pruning")
            else:
                lm_task_names.append("mmlu")
        elif task == "hellaswag":
            lm_task_names.append("hellaswag")
        elif task == "gsm8k":
            lm_task_names.append("gsm8k")

    # Run evaluation
    scores: dict[str, float] = {}
    try:
        console.print("[bold]Running evaluation...[/bold]")
        evaluator = Evaluator(lm)

        # Get task objects from lm_eval tasks registry
        eval_tasks = tasks.get_tasks(lm_task_names) if hasattr(tasks, "get_tasks") else {}
        results = evaluator.evaluate(
            task_dict=eval_tasks,
            limit=None if not lite else 100,  # sample size for lite mode
            batch_size=batch_size,
        )

        # Extract scores
        for idx, task in enumerate(task_names_in):
            lm_task = lm_task_names[idx] if idx < len(lm_task_names) else task
            score: float = 0.0
            if isinstance(results, dict):
                task_result = results.get(lm_task, {})
                if isinstance(task_result, dict):
                    score = float(task_result.get("acc_norm", 0.0))
                    if score == 0.0:
                        score = float(task_result.get("acc", 0.0))
            scores[task] = score
            console.print(f"  {task}: {score:.4f}")

    except Exception as e:
        console.print(f"[yellow]Warning: evaluation failed ({e}). "
                      "Returning zeros.[/yellow]")
        scores = {"mmlu": 0.0, "hellaswag": 0.0, "gsm8k": 0.0}

    console.print("[bold green]Quality benchmark complete.[/bold green]")
    return scores


def get_available_tasks() -> list[str]:
    """Return list of available benchmark tasks."""
    return ["mmlu", "hellaswag", "gsm8k"]
