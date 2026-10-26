"""Tests for fairbench.quality module."""

from unittest.mock import MagicMock, patch

from fairbench.quality import (
    AVAILABLE_BENCHMARKS,
    get_lite_benchmarks,
    list_benchmarks,
    run_quality_benchmark,
)


class TestListBenchmarks:
    def test_returns_all_benchmarks(self):
        """list_benchmarks returns all 8 benchmark categories."""
        benchmarks = list_benchmarks()
        assert len(benchmarks) == len(AVAILABLE_BENCHMARKS)
        for name in AVAILABLE_BENCHMARKS:
            assert name in benchmarks

    def test_returns_dict(self):
        """list_benchmarks returns a dict."""
        benchmarks = list_benchmarks()
        assert isinstance(benchmarks, dict)

    def test_all_benchmarks_have_required_keys(self):
        """Every benchmark has name, description, and lite keys."""
        benchmarks = list_benchmarks()
        for name, info in benchmarks.items():
            assert "name" in info, f"Missing 'name' for {name}"
            assert "description" in info, f"Missing 'description' for {name}"
            assert "lite" in info, f"Missing 'lite' for {name}"


class TestGetLiteBenchmarks:
    def test_returns_list(self):
        """get_lite_benchmarks returns a list."""
        lite = get_lite_benchmarks()
        assert isinstance(lite, list)

    def test_all_lite_benchmarks_are_valid(self):
        """All returned lite benchmarks exist in AVAILABLE_BENCHMARKS."""
        lite = get_lite_benchmarks()
        for name in lite:
            assert name in AVAILABLE_BENCHMARKS
            assert AVAILABLE_BENCHMARKS[name]["lite"] is True


class TestRunQualityBenchmark:
    def test_returns_list_of_results(self):
        """run_quality_benchmark returns a list of QualityResult."""
        # Mock simple_evaluate to return a valid result
        fake_result = {
            "results": {"mmlu": {"acc": 0.75, "items": 100, "correct": 75}},
        }
        fake_lm_eval = MagicMock()
        fake_lm_eval.simple_evaluate.return_value = fake_result
        with patch.dict("sys.modules", {"lm_eval": fake_lm_eval}):
            result = run_quality_benchmark(
                model_path="/nonexistent/model.gguf",
                benchmark_name="mmlu",
            )
            assert isinstance(result, list)
            assert len(result) > 0
            # Verify the score was parsed correctly
            assert result[0].score == 0.75

    def test_graceful_degradation_without_lm_eval(self):
        """run_quality_benchmark returns zero scores when lm_eval is unavailable."""
        # Mock the import inside the quality module
        with patch.dict("sys.modules", {"lm_eval": None}):
            result = run_quality_benchmark(
                model_path="/nonexistent/model.gguf",
                benchmark_name="mmlu",
            )
            for r in result:
                assert r.score == 0.0
                assert "fallback" in r.metadata

    def test_all_benchmarks_when_no_benchmark_specified(self):
        """run_quality_benchmark runs all benchmarks when no benchmark_name given."""
        with patch.dict("sys.modules", {"lm_eval": None}):
            result = run_quality_benchmark(
                model_path="/nonexistent/model.gguf",
            )
            assert len(result) == len(AVAILABLE_BENCHMARKS)
