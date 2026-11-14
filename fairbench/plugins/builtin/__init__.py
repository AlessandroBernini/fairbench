"""Built-in benchmark plugins for FairBench."""

from __future__ import annotations

from fairbench.plugins.base import BenchmarkPlugin, PluginRegistry


def _register_mmlu() -> None:
    """Register MMLU plugin."""

    class MMLUBenchmark(BenchmarkPlugin):
        name = "mmlu"
        version = "1.0"
        description = "Evaluates models on 57 tasks including STEM, humanities, and social sciences"
        category = "quality"

        def run(self, model_path: str, config: dict) -> dict:
            """Run MMLU benchmark using lm-eval."""
            try:
                import lm_eval

                results = lm_eval.simple_evaluate(
                    model="hf_auto",
                    model_args=f"pretrained={model_path}",
                    tasks=config.get("tasks", ["mmlu"]),
                    batch_size=config.get("batch_size", "auto"),
                    limit=config.get("limit", None),
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

    PluginRegistry.register(MMLUBenchmark())


def _register_hellaswag() -> None:
    """Register HellaSwag plugin."""

    class HellaSwagBenchmark(BenchmarkPlugin):
        name = "hellaswag"
        version = "1.0"
        description = "Evaluates commonsense reasoning and action recognition"
        category = "quality"

        def run(self, model_path: str, config: dict) -> dict:
            """Run HellaSwag benchmark."""
            try:
                import lm_eval

                results = lm_eval.simple_evaluate(
                    model="hf_auto",
                    model_args=f"pretrained={model_path}",
                    tasks=["hellaswag"],
                    batch_size=config.get("batch_size", "auto"),
                    limit=config.get("limit", None),
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

    PluginRegistry.register(HellaSwagBenchmark())


def _register_gsm8k() -> None:
    """Register GSM8K plugin."""

    class GSM8KBenchmark(BenchmarkPlugin):
        name = "gsm8k"
        version = "1.0"
        description = "Evaluates grade school math problem solving"
        category = "quality"

        def run(self, model_path: str, config: dict) -> dict:
            """Run GSM8K benchmark."""
            try:
                import lm_eval

                results = lm_eval.simple_evaluate(
                    model="hf_auto",
                    model_args=f"pretrained={model_path}",
                    tasks=["gsm8k"],
                    batch_size=config.get("batch_size", "auto"),
                    limit=config.get("limit", None),
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

    PluginRegistry.register(GSM8KBenchmark())


def _register_humaneval() -> None:
    """Register HumanEval plugin."""

    class HumanEvalBenchmark(BenchmarkPlugin):
        name = "humaneval"
        version = "1.0"
        description = "Evaluates code generation on hand-written Python problems"
        category = "quality"

        def run(self, model_path: str, config: dict) -> dict:
            """Run HumanEval benchmark."""
            try:
                import lm_eval

                results = lm_eval.simple_evaluate(
                    model="hf_auto",
                    model_args=f"pretrained={model_path}",
                    tasks=["humaneval"],
                    batch_size=config.get("batch_size", "auto"),
                    limit=config.get("limit", None),
                )

                return {
                    "results": results.get("results", {}),
                    "versions": results.get("versions", {}),
                    "config": results.get("config", {}),
                }
            except ImportError as e:
                raise RuntimeError(f"HumanEval benchmark requires lm-eval: {e}") from e

        def get_metadata(self) -> dict[str, str | bool]:
            """Return HumanEval metadata."""
            return {
                "name": "HumanEval",
                "description": self.description,
                "lite": True,
            }

    PluginRegistry.register(HumanEvalBenchmark())


# Register all built-in plugins on import
_register_mmlu()
_register_hellaswag()
_register_gsm8k()
_register_humaneval()
