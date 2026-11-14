"""HellaSwag benchmark plugin."""

from __future__ import annotations

from typing import Any

from fairbench.plugins.base import BenchmarkPlugin


class HellaSwagBenchmark(BenchmarkPlugin):
    """HellaSwag benchmark for commonsense reasoning."""

    name = "hellaswag"
    version = "1.0"
    description = "Evaluates commonsense reasoning and action recognition"
    category = "quality"

    def __init__(self) -> None:
        super().__init__()

    def run(self, model_path: str, config: dict) -> dict[str, Any]:
        """Run HellaSwag benchmark."""
        try:
            import lm_eval

            results = lm_eval.simple_evaluate(
                model="hf_auto",
                model_args=f"pretrained={model_path}",
                tasks=["hellaswag"],
                batch_size=config.get('batch_size', 'auto'),
                limit=config.get('limit', None),
            )

            return {
                "results": results.get("results", {}),
                "versions": results.get("versions", {}),
                "config": results.get("config", {}),
            }
        except ImportError as e:
            raise RuntimeError(f"HellaSwag benchmark requires lm-eval: {e}") from e

    def get_metadata(self) -> dict[str, str | bool]:
        """Return HellaSwag metadata."""
        return {
            "name": "HellaSwag",
            "description": self.description,
            "lite": True,
        }
