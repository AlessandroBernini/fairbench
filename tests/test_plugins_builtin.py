"""Smoke tests for FairBench builtin plugins.

These are lightweight integration smoke tests that verify the plugin classes
can be instantiated and expose the expected API surface without invoking
lm_eval or actual model evaluation.
"""

import pytest

from fairbench.plugins.builtin.gsm8k import GSM8KBenchmark
from fairbench.plugins.builtin.hellaswag import HellaSwagBenchmark
from fairbench.plugins.builtin.humaneval import HumanEvalBenchmark
from fairbench.plugins.builtin.mmlu import MMLUBenchmark


@pytest.mark.parametrize(
    "plugin_cls,expected_name",
    [
        (GSM8KBenchmark, "gsm8k"),
        (HellaSwagBenchmark, "hellaswag"),
        (HumanEvalBenchmark, "humaneval"),
        (MMLUBenchmark, "mmlu"),
    ],
)
def test_plugin_instantiation_and_metadata(plugin_cls, expected_name):
    plugin = plugin_cls()
    # Basic attributes
    assert plugin.name == expected_name
    assert hasattr(plugin, "version")
    assert hasattr(plugin, "description")
    assert hasattr(plugin, "category")

    # Metadata method returns dict with required keys
    meta = plugin.get_metadata()
    assert isinstance(meta, dict)
    assert "name" in meta
    assert "description" in meta

    # run method exists and is callable
    assert callable(getattr(plugin, "run", None))


def test_gsm8k_run_raises_without_lm_eval():
    plugin = GSM8KBenchmark()
    # Run should raise RuntimeError about missing lm-eval when import fails
    with pytest.raises(RuntimeError) as exc:
        plugin.run("/nonexistent/model.gguf", {})
    assert "lm-eval" in str(exc.value).lower()


def test_hellaswag_metadata_lite_flag():
    plugin = HellaSwagBenchmark()
    meta = plugin.get_metadata()
    assert meta.get("lite") is True


def test_humaneval_category():
    plugin = HumanEvalBenchmark()
    assert plugin.category == "quality"


def test_mmlu_version_present():
    plugin = MMLUBenchmark()
    assert plugin.version == "1.0"
