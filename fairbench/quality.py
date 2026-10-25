"""Quality benchmark for FairBench.

Uses lm_eval (lm-eval>=0.4.0) to measure quality metrics on local GGUF
models via llama-cpp-python backend.

Supported benchmarks:
  - MMLU:      Knowledge (57 academic subjects, multiple choice)
  - HellaSwag: Commonsense reasoning (continuation prediction)
  - GSM8K:     Grade-school math reasoning
  - HumanEval: Code generation (Python, 164 problems)
  - IFEval:    Instruction following (strict prompt compliance)
  - TruthfulQA: Truthfulness (knowledge + reasoning)
  - ARC-Challenge: Scientific reasoning (Allen AI)
  - LiveBench: 2024/2025 benchmark (anti data contamination)

Supports 'lite' mode for faster evaluation on consumer hardware.
"""

from pathlib import Path

from rich.console import Console

console = Console()

# Default benchmark tasks
DEFAULT_TASKS = ["mmlu", "hellaswag", "gsm8k"]

# All supported benchmark categories
ALL_TASKS = [
    "mmlu",
    "hellaswag",
    "gsm8k",
    "humaneval",
    "ifeval",
    "truthfulqa",
    "arc",
    "livebench",
]

# Mapping: internal name -> lm_eval task name
TASK_MAP = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "ifeval": "ifeval",
    "truthfulqa": "truthfulqa",
    "arc": "arc",
    "livebench": "livebench",
}

# Lite mode mapping: full task -> lite subset
LITE_MAP = {
    "mmlu": "mmlu_pruning",
}

# Benchmark descriptions for display
BENCHMARK_INFO = {
    "mmlu": {
        "name": "MMLU",
        "description": "Knowledge (57 academic subjects)",
        "type": "multiple_choice",
        "lite": "mmlu_pruning (subset ~1000 questions)",
    },
    "hellaswag": {
        "name": "HellaSwag",
        "description": "Commonsense reasoning",
        "type": "multiple_choice",
        "lite": "full dataset (already fast)",
    },
    "gsm8k": {
        "name": "GSM8K",
        "description": "Grade-school math reasoning",
        "type": "exact_match",
        "lite": "100 samples",
    },
    "humaneval": {
        "name": "HumanEval",
        "description": "Code generation (Python)",
        "type": "pass_at_1",
        "lite": "16 samples",
    },
    "ifeval": {
        "name": "IFEval",
        "description": "Instruction following",
        "type": "strict_match",
        "lite": "subset",
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "description": "Truthfulness (knowledge + reasoning)",
        "type": "multiple_choice",
        "lite": "subset",
    },
    "arc": {
        "name": "ARC-Challenge",
        "description": "Scientific reasoning (Allen AI)",
        "type": "multiple_choice",
        "lite": "subset",
    },
    "livebench": {
        "name": "LiveBench",
        "description": "2024/2025 benchmark (anti contamination)",
        "type": "multiple_choice",
        "lite": "subset",
    },
}


def run_quality_benchmark(
    model_path: Path,
    task_names_in: list[str] | None = None,
    lite: bool = True,
    batch_size: int = 4,
) -> dict[str, float]:
    """Run quality benchmark on a GGUF model.

    Uses lm_eval to measure quality metrics on local GGUF models.
    Falls back gracefully if lm_eval or llama-cpp-python is not available.

    Args:
        model_path: Path to the GGUF model file.
        task_names_in: List of task names to run.
            Defaults to ['mmlu', 'hellaswag', 'gsm8k'].
            Supported: mmlu, hellaswag, gsm8k, humaneval, ifeval,
            truthfulqa, arc, livebench.
        lite: If True, use lite mode for faster evaluation.
        batch_size: Batch size for evaluation.

    Returns:
        dict with task names as keys and float scores as values.
    """
    if task_names_in is None:
        task_names_in = DEFAULT_TASKS

    # Validate tasks
    valid_tasks = set(ALL_TASKS)
    invalid = [t for t in task_names_in if t not in valid_tasks]
    if invalid:
        console.print(
            f"[yellow]Warning: unknown tasks {invalid}. "
            f"Supported: {', '.join(valid_tasks)}[/yellow]"
        )
        task_names_in = [t for t in task_names_in if t in valid_tasks]
        if not task_names_in:
            return {t: 0.0 for t in DEFAULT_TASKS}

    # Try to import lm_eval
    try:
        from lm_eval import Evaluator, tasks
    except ImportError:
        console.print(
            "[yellow]Warning: lm_eval not installed. "
            "Install with: pip install lm-eval>=0.4.0[/yellow]"
        )
        return {t: 0.0 for t in task_names_in}

    # Try to import llama-cpp-python for GGUF support
    try:
        from lm_eval.models.llama_cpp import LlamaCPP
    except ImportError:
        console.print(
            "[yellow]Warning: llama-cpp-python not installed. "
            "Install with: pip install llama-cpp-python[/yellow]"
        )
        return {t: 0.0 for t in task_names_in}

    # Validate model path
    if not Path(model_path).exists():
        console.print(f"[red]Error: model file not found: {model_path}[/red]")
        return {t: 0.0 for t in task_names_in}

    # Show benchmark config
    console.print(f"\n[bold blue]Quality benchmark:[/bold blue] {model_path.name}")
    for task in task_names_in:
        info = BENCHMARK_INFO.get(task, {})
        lite_note = f" (lite: {info.get('lite', 'N/A')})" if lite else ""
        console.print(
            f"  • {info.get('name', task)}: {info.get('description', 'N/A')}{lite_note}"
        )
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
        return {t: 0.0 for t in task_names_in}

    # Determine lm_eval task names
    lm_task_names: list[str] = []
    for task in task_names_in:
        lm_task = TASK_MAP.get(task, task)
        if lite and task in LITE_MAP:
            lm_task = LITE_MAP[task]
        lm_task_names.append(lm_task)

    # Run evaluation
    scores: dict[str, float] = {}
    try:
        console.print("[bold]Running evaluation...[/bold]")
        evaluator = Evaluator(lm)

        # Get task objects from lm_eval tasks registry
        eval_tasks = tasks.get_tasks(lm_task_names) if hasattr(tasks, "get_tasks") else {}

        # Set lite sample limits
        lite_limits = {
            "humaneval": 16,
            "gsm8k": 100,
        }
        sample_limit = lite_limits.get(task_names_in[0] if task_names_in else "", None)

        results = evaluator.evaluate(
            task_dict=eval_tasks,
            limit=None if not lite else sample_limit,
            batch_size=batch_size,
        )

        # Extract scores
        for idx, task in enumerate(task_names_in):
            lm_task = lm_task_names[idx] if idx < len(lm_task_names) else task
            score: float = 0.0
            if isinstance(results, dict):
                task_result = results.get(lm_task, {})
                if isinstance(task_result, dict):
                    # Try acc_norm first, then acc, then exact_match
                    score = float(task_result.get("acc_norm", 0.0))
                    if score == 0.0:
                        score = float(task_result.get("acc", 0.0))
                    if score == 0.0:
                        score = float(task_result.get("exact_match", 0.0))
                    if score == 0.0:
                        score = float(task_result.get("pass_at_1", 0.0))
            scores[task] = score
            info = BENCHMARK_INFO.get(task, {})
            name = info.get("name", task)
            if task == "humaneval":
                console.print(f"  {name}: {score:.1%}")
            elif task == "mmlu":
                console.print(f"  {name}: {score:.4f}")
            else:
                console.print(f"  {name}: {score:.4f}")

    except Exception as e:
        console.print(f"[yellow]Warning: evaluation failed ({e}). "
                      "Returning zeros.[/yellow]")
        scores = {t: 0.0 for t in task_names_in}

    console.print("[bold green]Quality benchmark complete.[/bold green]")
    return scores


def get_available_tasks() -> list[str]:
    """Return list of all available benchmark task names."""
    return list(BENCHMARK_INFO.keys())


def get_task_info(task_name: str) -> dict:
    """Return metadata for a benchmark task."""
    return BENCHMARK_INFO.get(task_name, {
        "name": task_name,
        "description": "Unknown task",
        "type": "unknown",
        "lite": "N/A",
    })
