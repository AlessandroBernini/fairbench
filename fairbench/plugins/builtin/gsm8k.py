"""GSM8K benchmark plugin."""

from __future__ import annotations

from typing import Any

from fairbench.plugins.base import BenchmarkPlugin


class GSM8KBenchmark(BenchmarkPlugin):
    """GSM8K benchmark for grade school math."""

    name = "gsm8k"
    version = "1.0"
    description = "Evaluates grade school math problem solving"
    category = "quality"

    def __init__(self) -> None:
        super().__init__()

    def run(self, model_path: str, config: dict) -> dict[str, Any]:
        """Run GSM8K benchmark."""
        try:
            import lm_eval

            results = lm_eval.simple_evaluate(
                model="hf_auto",
                model_args=f"pretrained={model_path}",
                tasks=["gsm8k"],
                batch_size=config.get('batch_size', 'auto'),
                limit=config.get('limit', None),
            )

            return {
                "results": results.get("results", {}),
                "versions": results.get("versions", {}),
                "config": results.get("config", {}),
            }
        except ImportError as e:
            raise RuntimeError(f"GSM8K benchmark requires lm-eval: {e}") from e

    def get_metadata(self) -> dict[str, str | bool]:
        """Return GSM8K metadata."""
        return {
            "name": "GSM8K",
            "description": self.description,
            "lite": True,
        }
