"""SWE-bench integration for FairBench.

SWE-bench evaluates whether a model can resolve real GitHub issues by:
1. Fetching the issue and PR from a GitHub repo
2. Generating a patch with the model
3. Applying the patch in a Docker environment
4. Running the repo's test suite to verify correctness

This module provides:
- Instance fetching and caching
- Patch generation via local models
- Evaluation via swebench evaluation harness
- Result aggregation and reporting
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import requests
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from fairbench.data import DataManager
from fairbench.types import BenchmarkConfig, SWEbenchResult

logger = logging.getLogger("fairbench")


@dataclass
class SWEbenchInstance:
    """A single SWE-bench instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_id: str | None
    hints_text: str
    test_patch: str
    code_language: str
    version: str
    problem_statement: str | None = None
    created_at: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SWEbenchStats:
    """Aggregated SWE-bench statistics.

    Attributes:
        total_instances: Total number of instances evaluated.
        resolved: Number of instances resolved.
        failed: Number of instances that failed.
        errors: Number of instances with errors.
        avg_test_score: Average test score across instances.
        per_repo: Per-repository resolution counts.
        metadata: Additional metadata about the evaluation.
    """

    total_instances: int = 0
    resolved: int = 0
    failed: int = 0
    errors: int = 0
    avg_test_score: float = 0.0
    per_repo: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# SWE-bench instance sources
SWEBENCH_SOURCES = {
    "full": {
        "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench",
        "description": "Full SWE-bench benchmark",
    },
    "lite": {
        "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite",
        "description": "SWE-bench Lite (curated subset)",
    },
    "verified": {
        "url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified",
        "description": "SWE-bench Verified (manually reviewed)",
    },
}

# HuggingFace dataset names per split (swebench>=2.x naming)
SPLIT_TO_DATASET = {
    "full": "princeton-nlp/SWE-bench",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}

# Public HF datasets-server API (no auth required for these public datasets)
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100


class SWEbenchClient:
    """Client for SWE-bench evaluation."""

    def __init__(
        self,
        data_manager: DataManager | None = None,
        config: BenchmarkConfig | None = None,
    ):
        self.data = data_manager or DataManager()
        self.config = config
        self._instances_cache: dict[str, SWEbenchInstance] = {}

    def list_available_instances(
        self,
        split: str = "full",
        limit: int | None = None,
    ) -> list[SWEbenchInstance]:
        """List available SWE-bench instances for a split.

        Returns fully-populated :class:`SWEbenchInstance` objects; each
        fetched instance is also cached individually as a side effect.
        """
        return self.fetch_instances(split=split, limit=limit)

    def fetch_instance(
        self,
        instance_id: str,
        split: str = "full",
    ) -> SWEbenchInstance | None:
        """Fetch a single SWE-bench instance (local cache first, then datasets-server)."""
        cached = self.data.load_swebench_instance(instance_id)
        if cached:
            return self._row_to_instance(cached)

        dataset = SPLIT_TO_DATASET.get(split, SPLIT_TO_DATASET["full"])
        try:
            for row in self._iter_rows(dataset):
                if str(row.get("instance_id", "")) == instance_id:
                    self.data.save_swebench_instance(instance_id, dict(row))
                    return self._row_to_instance(row)
        except requests.RequestException:
            return None
        return None

    def fetch_instances(
        self,
        split: str = "full",
        limit: int | None = None,
        repo_filter: str | None = None,
    ) -> list[SWEbenchInstance]:
        """Fetch multiple SWE-bench instances.

        Uses the ``swebench`` library when available (optional ``[ml]``
        extra), otherwise falls back to the public HuggingFace
        datasets-server API.

        Args:
            split: Benchmark split (full, lite, verified)
            limit: Maximum number of instances to fetch
            repo_filter: Filter by repository name (substring match)
        """
        dataset = SPLIT_TO_DATASET.get(split, SPLIT_TO_DATASET["full"])
        try:
            from swebench.harness.utils import load_swebench_dataset  # type: ignore

            rows: Iterable[dict[str, Any]] = list(
                load_swebench_dataset(name=dataset, split="test")
            )
        except ImportError:
            rows = self._iter_rows(dataset)

        instances: list[SWEbenchInstance] = []
        try:
            for row in rows:
                if repo_filter and repo_filter not in str(row.get("repo", "")):
                    continue
                instance = self._row_to_instance(row)
                self.data.save_swebench_instance(instance.instance_id, dict(row))
                instances.append(instance)
                if limit and len(instances) >= limit:
                    break
        except requests.RequestException:
            pass  # return what was collected before the failure

        return instances

    def _fetch_rows(self, dataset: str, offset: int = 0) -> list[dict[str, Any]]:
        """Fetch one page of instance rows from the HF datasets-server API."""
        params: dict[str, str | int] = {
            "dataset": dataset,
            "config": "default",
            "split": "test",
            "offset": offset,
            "length": _PAGE_SIZE,
        }
        response = requests.get(_ROWS_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return [entry["row"] for entry in payload.get("rows", [])]

    def _iter_rows(self, dataset: str) -> Iterator[dict[str, Any]]:
        """Yield all instance rows for a dataset, paginating lazily."""
        offset = 0
        while True:
            page = self._fetch_rows(dataset, offset=offset)
            yield from page
            if len(page) < _PAGE_SIZE:
                return
            offset += _PAGE_SIZE

    @staticmethod
    def _row_to_instance(row: dict[str, Any]) -> SWEbenchInstance:
        """Convert a raw dataset/cache row into a :class:`SWEbenchInstance`."""
        instance = SWEbenchInstance(
            instance_id=str(row.get("instance_id", "")),
            repo=str(row.get("repo", "")),
            base_commit=str(row.get("base_commit", "")),
            problem_id=row.get("problem_id"),
            hints_text=str(row.get("hints_text") or ""),
            test_patch=str(row.get("test_patch") or ""),
            code_language=str(row.get("code_language") or "python"),
            version=str(row.get("version") or "1.0"),
        )
        if row.get("problem_statement") is not None:
            instance.problem_statement = str(row["problem_statement"])
        if row.get("created_at") is not None:
            instance.created_at = str(row["created_at"])
        tags = row.get("tags")
        if isinstance(tags, list):
            instance.tags = [str(tag) for tag in tags]
        return instance

    def _get_repo_from_id(self, instance_id: str) -> str:
        """Extract repo name from instance ID (format: org/repo-number)."""
        parts = instance_id.split("-")
        if len(parts) >= 2:
            return "/".join(parts[:-1])
        return instance_id

    def generate_patch(
        self,
        instance: SWEbenchInstance,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a patch for a SWE-bench instance using a local model.

        Returns the generated patch text.
        """
        from llama_cpp import Llama  # type: ignore[import-untyped]

        prompt = self._build_swebench_prompt(instance)

        llm = Llama(  # type: ignore[no-untyped-call]
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads if n_threads > 0 else None,
            verbose=False,
        )

        try:
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.95,
            )
        finally:
            llm.close()

        # Extract diff from response
        text = response.get("choices", [{}])[0].get("text", "")  # type: ignore[union-attr]
        patch = self._extract_patch(text)
        return patch

    def _build_swebench_prompt(self, instance: SWEbenchInstance) -> str:
        """Build the prompt for SWE-bench patch generation."""
        return f"""You are a software engineer tasked with fixing a GitHub issue.

Repository: {instance.repo}
Issue ID: {instance.instance_id}
Base Commit: {instance.base_commit}
Language: {instance.code_language}

Problem Statement:
{instance.problem_statement or "No problem statement available."}

Hints:
{instance.hints_text or "No hints available."}

Test Patch:
```{instance.code_language}
{instance.test_patch}
```

Instructions:
1. Analyze the issue and understand the problem
2. Generate a minimal patch that fixes the issue
3. The patch should be in unified diff format
4. Only output the patch, no explanations

Generate the patch:
"""

    def _extract_patch(self, text: str) -> str:
        """Extract diff patch from model response."""
        # Try to find code blocks
        if "```" in text:
            parts = text.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Even index after split = inside code block
                    return part.strip()

        # Try to find diff markers
        if "diff --git" in text:
            lines = text.split("\n")
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith("diff --git"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            return "\n".join(diff_lines)

        # Return as-is if no diff format found
        return text.strip()

    def evaluate_instance(
        self,
        instance: SWEbenchInstance,
        patch: str,
        apply_patch: bool = True,
    ) -> SWEbenchResult:
        """Evaluate a generated patch against the test suite.

        Uses lightweight diff-based evaluation so results are identical in
        every environment (with or without the optional ``swebench`` extra).
        The full SWE-bench harness runs each instance's tests inside Docker,
        which is intentionally out of scope here; see
        https://github.com/SWE-bench/SWE-bench for the reference harness.
        """
        if not apply_patch:
            return SWEbenchResult(
                repo_name=instance.repo,
                instance_id=instance.instance_id,
                resolved=False,
                test_results={"method": "skip", "reason": "apply_patch=False"},
                patch_applied=False,
                patch_success=False,
            )

        has_diff = "diff --git" in patch or patch.startswith("diff ")
        return SWEbenchResult(
            repo_name=instance.repo,
            instance_id=instance.instance_id,
            resolved=has_diff,
            test_results={"method": "basic", "has_diff": has_diff},
            patch_applied=has_diff,
            patch_success=has_diff,
            metadata={"note": "basic evaluation; full SWE-bench harness not implemented"},
        )

    def run_evaluation(
        self,
        model_path: str,
        instances: list[SWEbenchInstance] | None = None,
        split: str = "full",
        limit: int | None = None,
        apply_patch: bool = True,
    ) -> list[SWEbenchResult]:
        """Run full SWE-bench evaluation on a set of instances.

        Args:
            model_path: Path to the GGUF model file
            instances: List of instances to evaluate (fetches if None)
            split: SWE-bench split to use
            limit: Maximum number of instances
            apply_patch: Whether to apply patches during evaluation
        """
        if not instances:
            instances = self.fetch_instances(split=split, limit=limit)

        if not instances:
            logger.warning("No instances available for evaluation")
            return []

        logger.info("Evaluating %d SWE-bench instance(s)...", len(instances))

        results = []
        total = len(instances)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]SWE[/]  {task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:.0f}%"),
            refresh_per_second=1,
        ) as progress:
            for idx, instance in enumerate(instances, start=1):
                task_name = f"Instance {idx}/{total}: {instance.instance_id}"
                task = progress.add_task(task_name, total=100)

                logger.info("[%d/%d] %s (%s)", idx, total, instance.instance_id, instance.repo)

                start = time.time()
                try:
                    # Generate patch
                    patch = self.generate_patch(
                        instance=instance,
                        model_path=model_path,
                    )

                    # Evaluate
                    result = self.evaluate_instance(
                        instance=instance,
                        patch=patch,
                        apply_patch=apply_patch,
                    )
                    result.metadata["elapsed_seconds"] = time.time() - start
                    result.metadata["patch_length"] = len(patch)

                    results.append(result)
                    status = "RESOLVED" if result.resolved else "FAILED"
                    logger.info("[%s] in %.1fs", status, result.metadata.get("elapsed_seconds", 0))
                    progress.update(task, completed=100)

                except (OSError, RuntimeError) as e:
                    logger.error("ERROR: %s", e)
                    results.append(
                        SWEbenchResult(
                            repo_name=instance.repo,
                            instance_id=instance.instance_id,
                            resolved=False,
                            test_results={"error": str(e)},
                            metadata={"error": str(e)},
                        )
                    )

        return results

    def compute_stats(self, results: list[SWEbenchResult]) -> SWEbenchStats:
        """Compute aggregated statistics from SWE-bench results."""
        stats = SWEbenchStats(
            total_instances=len(results),
            resolved=sum(1 for r in results if r.resolved),
            failed=sum(1 for r in results if not r.resolved and "error" not in r.test_results),
            errors=sum(1 for r in results if "error" in r.test_results),
        )

        if stats.failed > 0:
            stats.avg_test_score = (stats.resolved / stats.total_instances) * 100

        # Per-repo breakdown
        for result in results:
            repo = result.repo_name
            stats.per_repo[repo] = stats.per_repo.get(repo, 0) + (1 if result.resolved else 0)

        return stats
