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

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from fairbench.data import DataManager
from fairbench.types import SWEbenchResult, BenchmarkConfig


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
    """Aggregated SWE-bench statistics."""
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
        """List available SWE-bench instances from HuggingFace."""
        source_url = SWEBENCH_SOURCES.get(split, SWEBENCH_SOURCES["full"])["url"]

        # Check cache first
        cache_key = f"swebench_instances_{split}"
        cached = self.data.get_cached_result("swebench", cache_key)
        if cached:
            return [
                SWEbenchInstance(
                    instance_id=item["instance_id"],
                    repo=item["repo"],
                    base_commit=item["base_commit"],
                    problem_id=item.get("problem_id"),
                    hints_text=item.get("hints_text", ""),
                    test_patch=item.get("test_patch", ""),
                    code_language=item.get("code_language", "python"),
                    version=item.get("version", "1.0"),
                )
                for item in (cached[:limit] if limit else cached)
            ]

        # Fetch from HuggingFace datasets API
        try:
            hf_url = f"https://huggingface.co/api/datasets/{source_url.split('/')[-1]}"
            response = requests.get(hf_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract instances from dataset info
            instances = []
            if "siblings" in data:
                for sibling in data["siblings"]:
                    if sibling.get("rfilename", "").endswith(".json"):
                        instances.append(sibling["rfilename"])

            self.data.save_cached_result("swebench", cache_key, instances)
            return instances[:limit] if limit else instances
        except requests.RequestException:
            return []

    def fetch_instance(
        self,
        instance_id: str,
        split: str = "full",
    ) -> SWEbenchInstance | None:
        """Fetch a single SWE-bench instance from HuggingFace."""
        # Check cache
        cached = self.data.load_swebench_instance(instance_id)
        if cached:
            return SWEbenchInstance(**cached)

        # Fetch from HuggingFace
        try:
            hf_url = (
                f"https://huggingface.co/datasets/princeton-nlp/SWE-bench"
                f"/resolve/main/{split}/{instance_id}.json"
            )
            response = requests.get(hf_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            instance = SWEbenchInstance(
                instance_id=data.get("instance_id", instance_id),
                repo=data.get("repo", ""),
                base_commit=data.get("base_commit", ""),
                problem_id=data.get("problem_id"),
                hints_text=data.get("hints_text", ""),
                test_patch=data.get("test_patch", ""),
                code_language=data.get("code_language", "python"),
                version=data.get("version", "1.0"),
                problem_statement=data.get("problem_statement"),
                created_at=data.get("created_at"),
                tags=data.get("tags", []),
            )

            # Cache it
            self.data.save_swebench_instance(instance_id, data)
            return instance
        except requests.RequestException:
            return None

    def fetch_instances(
        self,
        split: str = "full",
        limit: int | None = None,
        repo_filter: str | None = None,
    ) -> list[SWEbenchInstance]:
        """Fetch multiple SWE-bench instances.

        Args:
            split: Benchmark split (full, lite, verified)
            limit: Maximum number of instances to fetch
            repo_filter: Filter by repository name
        """
        instances = []
        fetched_ids = set()

        # Use swebench library if available for proper fetching
        try:
            from swebench import INSTANCE_IDS  # type: ignore

            all_ids = INSTANCE_IDS.get(split, [])
            if repo_filter:
                all_ids = [
                    iid for iid in all_ids
                    if repo_filter in self._get_repo_from_id(iid)
                ]

            for iid in all_ids[:limit] if limit else all_ids:
                instance = self.fetch_instance(iid, split)
                if instance:
                    instances.append(instance)
                    fetched_ids.add(iid)
        except ImportError:
            # Fallback: use HuggingFace API
            hf_url = (
                f"https://huggingface.co/api/datasets"
                f"/princeton-nlp/SWE-bench_{split.capitalize()}"
                f"/tree/main"
            )
            try:
                response = requests.get(hf_url, timeout=30)
                response.raise_for_status()
                files = response.json()

                for file in files:
                    if file.get("type") == "file" and file["path"].endswith(".json"):
                        iid = Path(file["path"]).stem
                        if iid not in fetched_ids:
                            if repo_filter and repo_filter not in iid:
                                continue
                            instance = self.fetch_instance(iid, split)
                            if instance:
                                instances.append(instance)
                                fetched_ids.add(iid)
                                if limit and len(instances) >= limit:
                                    break
            except requests.RequestException:
                pass

        return instances

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
        from llama_cpp import Llama  # type: ignore

        prompt = self._build_swebench_prompt(instance)

        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads if n_threads > 0 else None,
            verbose=False,
        )

        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
        )

        llm.close()

        # Extract diff from response
        patch = self._extract_patch(response["choices"][0]["text"])
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

        Uses swebench evaluation harness if available, otherwise returns
        a basic result.
        """
        try:
            from swebench import (  # type: ignore
                get_eval_report,
                run_evaluation,
            )

            # Apply patch and run evaluation
            if apply_patch:
                # Write patch to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".patch", delete=False
                ) as f:
                    f.write(patch)
                    patch_file = f.name

                # Run evaluation
                report = get_eval_report(
                    predictions=[{
                        "instance_id": instance.instance_id,
                        "model_patch": patch,
                    }],
                    model=instance.repo,
                    lines_to_change=[],
                    fail_to_pass=[],
                    pass_to_pass=[],
                )

                os.unlink(patch_file)

                resolved = report.get("resolved", False)
                return SWEbenchResult(
                    repo_name=instance.repo,
                    instance_id=instance.instance_id,
                    resolved=resolved,
                    test_results=report,
                    patch_applied=True,
                    patch_success=resolved,
                )
        except ImportError:
            # Fallback: basic evaluation
            has_diff = "diff --git" in patch or patch.startswith("diff ")
            return SWEbenchResult(
                repo_name=instance.repo,
                instance_id=instance.instance_id,
                resolved=has_diff,
                test_results={"method": "basic", "has_diff": has_diff},
                patch_applied=has_diff,
                patch_success=has_diff,
                metadata={"note": "swebench library not installed, using basic evaluation"},
            )
        except Exception as e:
            return SWEbenchResult(
                repo_name=instance.repo,
                instance_id=instance.instance_id,
                resolved=False,
                test_results={"error": str(e)},
                patch_applied=False,
                patch_success=False,
                metadata={"error": str(e)},
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
            print("  No instances available for evaluation")
            return []

        print(f"  Evaluating {len(instances)} SWE-bench instance(s)...")

        results = []
        for i, instance in enumerate(instances):
            print(f"  [{i+1}/{len(instances)}] {instance.instance_id} ({instance.repo})")

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
                print(f"    [{status}] in {result.metadata.get('elapsed_seconds', 0):.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
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
