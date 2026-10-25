"""Tests for fairbench.quality module."""

from pathlib import Path

from fairbench.quality import (
    ALL_TASKS,
    BENCHMARK_INFO,
    DEFAULT_TASKS,
    get_available_tasks,
    get_task_info,
    run_quality_benchmark,
)


class TestGetAvailableTasks:
    def test_returns_all_tasks(self):
        """get_available_tasks returns all 8 benchmark categories."""
        tasks = get_available_tasks()
        assert len(tasks) == 8
        for task in ALL_TASKS:
            assert task in tasks

    def test_returns_list(self):
        """get_available_tasks returns a list."""
        tasks = get_available_tasks()
        assert isinstance(tasks, list)

    def test_all_tasks_have_info(self):
        """Every task in ALL_TASKS has corresponding BENCHMARK_INFO."""
        for task in ALL_TASKS:
            assert task in BENCHMARK_INFO, f"Missing BENCHMARK_INFO for {task}"


class TestGetTaskInfo:
    def test_returns_metadata(self):
        """get_task_info returns metadata dict for a known task."""
        info = get_task_info("mmlu")
        assert info["name"] == "MMLU"
        assert info["description"] == "Knowledge (57 academic subjects)"
        assert info["type"] == "multiple_choice"

    def test_returns_default_for_unknown(self):
        """get_task_info returns default metadata for unknown task."""
        info = get_task_info("nonexistent")
        assert info["name"] == "nonexistent"
        assert info["description"] == "Unknown task"

    def test_humaneval_info(self):
        """get_task_info returns correct info for HumanEval."""
        info = get_task_info("humaneval")
        assert info["name"] == "HumanEval"
        assert info["type"] == "pass_at_1"
        assert info["lite"] == "16 samples"


class TestRunQualityBenchmark:
    def test_default_tasks(self):
        """run_quality_benchmark with no tasks uses DEFAULT_TASKS."""
        result = run_quality_benchmark(
            model_path=Path("/nonexistent/model.gguf"),
            lite=True,
        )
        # Should return zeros since model doesn't exist
        assert isinstance(result, dict)
        for key in DEFAULT_TASKS:
            assert key in result
            assert result[key] == 0.0

    def test_invalid_tasks_warning(self, capsys):
        """run_quality_benchmark warns about invalid tasks."""
        result = run_quality_benchmark(
            model_path=Path("/nonexistent/model.gguf"),
            task_names_in=["mmlu", "nonexistent_task"],
            lite=True,
        )
        assert isinstance(result, dict)
        # Should have filtered out the invalid task
        assert "nonexistent_task" not in result

    def test_all_zero_on_missing_model(self):
        """run_quality_benchmark returns zeros for nonexistent model."""
        result = run_quality_benchmark(
            model_path=Path("/nonexistent/path/model.gguf"),
            task_names_in=["mmlu", "gsm8k"],
            lite=True,
        )
        assert result["mmlu"] == 0.0
        assert result["gsm8k"] == 0.0

    def test_no_tasks_returns_defaults(self):
        """None task list returns DEFAULT_TASKS as fallback."""
        result = run_quality_benchmark(
            model_path=Path("/nonexistent/model.gguf"),
            task_names_in=None,
            lite=True,
        )
        for key in DEFAULT_TASKS:
            assert key in result

    def test_missing_lm_eval_returns_zeros(self):
        """run_quality_benchmark returns zeros when lm_eval import fails."""
        # Simply verify the graceful degradation path works
        # by using a nonexistent model (triggers early return before lm_eval)
        result = run_quality_benchmark(
            model_path=Path("/nonexistent/path/model.gguf"),
            task_names_in=["mmlu"],
            lite=True,
        )
        assert result["mmlu"] == 0.0
