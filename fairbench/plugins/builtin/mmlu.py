"""MMLU benchmark plugin."""

from __future__ import annotations

from typing import Any

from fairbench.plugins.base import BenchmarkPlugin


class MMLUBenchmark(BenchmarkPlugin):
    """MMLU (Massive Multitask Language Understanding) benchmark."""

    name = "mmlu"
    version = "1.0"
    description = "Evaluates models on 57 tasks including STEM, humanities, and social sciences"
    category = "quality"

    def __init__(self) -> None:
        super().__init__()

    def run(self, model_path: str, config: dict) -> dict[str, Any]:
        """Run MMLU benchmark using lm-eval."""
        try:
            import lm_eval
            from lm_eval.api.model import PreTrainedModel

            # Check if we have a working model
            if not hasattr(PreTrainedModel, 'eval'):
                raise ImportError("lm_eval requires transformers or similar backend")

            tasks = config.get('tasks', ['mmlu'])
            batch_size = config.get('batch_size', 'auto')

            results = lm_eval.simple_evaluate(
                model="hf_auto",
                model_args=f"pretrained={model_path}",
                tasks=tasks,
                batch_size=batch_size,
                limit=config.get('limit', None),
            )

            return {
                "results": results.get("results", {}),
                "versions": results.get("versions", {}),
                "config": results.get("config", {}),
            }
        except ImportError as e:
            raise RuntimeError(f"MMLU benchmark requires lm-eval: {e}") from e

    def get_metadata(self) -> dict[str, str | bool]:
        """Return MMLU metadata."""
        return {
            "name": "MMLU",
            "description": self.description,
            "lite": True,
        }
