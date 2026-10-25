"""Tests for SWE-bench integration."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from fairbench.data import DataManager
from fairbench.swebench import (
    SWEBENCH_SOURCES,
    SWEbenchClient,
    SWEbenchInstance,
    SWEbenchStats,
)
from fairbench.types import BenchmarkConfig, SWEbenchResult


class TestSWEbenchInstance:
    """Tests for SWEbenchInstance dataclass."""

    def test_creation_with_required_fields(self):
        instance = SWEbenchInstance(
            instance_id="django-django-12345",
            repo="django/django",
            base_commit="abc123",
            problem_id="12345",
            hints_text="Some hints",
            test_patch="diff --git a/test.py",
            code_language="python",
            version="1.0",
        )
        assert instance.instance_id == "django-django-12345"
        assert instance.repo == "django/django"
        assert instance.problem_statement is None
        assert instance.tags == []

    def test_creation_with_all_fields(self):
        instance = SWEbenchInstance(
            instance_id="flask-flask-67890",
            repo="pallets/flask",
            base_commit="def456",
            problem_id="67890",
            hints_text="Hints here",
            test_patch="diff --git b/test.py",
            code_language="python",
            version="2.0",
            problem_statement="Fix the bug",
            created_at="2024-01-01",
            tags=["bug", "fix"],
        )
        assert instance.problem_statement == "Fix the bug"
        assert instance.created_at == "2024-01-01"
        assert instance.tags == ["bug", "fix"]


class TestSWEbenchStats:
    """Tests for SWEbenchStats dataclass."""

    def test_default_stats(self):
        stats = SWEbenchStats()
        assert stats.total_instances == 0
        assert stats.resolved == 0
        assert stats.failed == 0
        assert stats.errors == 0
        assert stats.avg_test_score == 0.0
        assert stats.per_repo == {}

    def test_compute_stats_from_results(self):
        results = [
            SWEbenchResult(
                repo_name="django/django",
                instance_id="django-django-1",
                resolved=True,
                test_results={"test_score": 1.0},
            ),
            SWEbenchResult(
                repo_name="django/django",
                instance_id="django-django-2",
                resolved=False,
                test_results={"test_score": 0.0},
            ),
            SWEbenchResult(
                repo_name="flask/flask",
                instance_id="flask-flask-1",
                resolved=True,
                test_results={"test_score": 1.0},
            ),
        ]

        client = SWEbenchClient()
        stats = client.compute_stats(results)

        assert stats.total_instances == 3
        assert stats.resolved == 2
        assert stats.failed == 1
        assert stats.errors == 0
        assert stats.avg_test_score == pytest.approx(66.67, abs=0.1)
        assert stats.per_repo == {"django/django": 1, "flask/flask": 1}


class TestSWEbenchClient:
    """Tests for SWEbenchClient."""

    @pytest.fixture
    def client(self):
        return SWEbenchClient()

    def test_list_available_splits(self):
        assert "full" in SWEBENCH_SOURCES
        assert "lite" in SWEBENCH_SOURCES
        assert "verified" in SWEBENCH_SOURCES
        assert SWEBENCH_SOURCES["full"]["description"] == "Full SWE-bench benchmark"
        assert SWEBENCH_SOURCES["lite"]["description"] == "SWE-bench Lite (curated subset)"

    def test_get_repo_from_id(self):
        client = SWEbenchClient()
        assert client._get_repo_from_id("django-django-12345") == "django/django"
        assert client._get_repo_from_id("flask-flask-67890") == "flask/flask"
        assert client._get_repo_from_id("singleword") == "singleword"

    def test_extract_patch_with_code_blocks(self):
        client = SWEbenchClient()
        text = "Here is the patch:\n```python\ndiff --git a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new\n```\nDone."
        patch = client._extract_patch(text)
        assert "diff --git" in patch
        assert "old" in patch
        assert "new" in patch

    def test_extract_patch_with_diff_markers(self):
        client = SWEbenchClient()
        text = "Let me fix this:\n\ndiff --git a/file.py b/file.py\nindex abc123..def456 100644\n--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old line\n+new line\n"
        patch = client._extract_patch(text)
        assert "diff --git" in patch

    def test_extract_patch_without_markers(self):
        client = SWEbenchClient()
        text = "Just some text without any diff markers"
        patch = client._extract_patch(text)
        assert patch == text.strip()


class TestDataManager:
    """Tests for DataManager with SWE-bench support."""

    def test_save_and_load_swebench_instance(self, tmp_path):
        dm = DataManager(str(tmp_path / "results"))
        instance_data = {
            "instance_id": "django-django-1",
            "repo": "django/django",
            "base_commit": "abc123",
        }

        filepath = dm.save_swebench_instance("django-django-1", instance_data)
        assert filepath.exists()

        loaded = dm.load_swebench_instance("django-django-1")
        assert loaded is not None
        assert loaded["instance_id"] == "django-django-1"

        non_existent = dm.load_swebench_instance("nonexistent")
        assert non_existent is None


class TestSWEbenchClientFetch:
    """Tests for SWEbenchClient fetch methods."""

    @pytest.fixture
    def client(self):
        return SWEbenchClient()

    def test_fetch_instance_fallback_without_swebench(self, client):
        """Test fetch_instance falls back gracefully when swebench not installed."""
        with (
            patch.object(client.data, "load_swebench_instance", return_value=None),
            patch("requests.get") as mock_get,
        ):
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.json.return_value = {
                "instance_id": "django-django-1",
                "repo": "django/django",
                "base_commit": "abc123",
                "hints_text": "",
                "test_patch": "",
                "code_language": "python",
                "version": "1.0",
            }

            instance = client.fetch_instance("django-django-1", split="full")
            assert instance is not None
            assert instance.instance_id == "django-django-1"

    def test_fetch_instance_returns_none_on_error(self, client):
        """Test fetch_instance returns None on network error."""
        with (
            patch.object(client.data, "load_swebench_instance", return_value=None),
            patch("requests.get") as mock_get,
        ):
            mock_get.side_effect = requests.RequestException("Network error")

            instance = client.fetch_instance("django-django-1", split="full")
            assert instance is None


class TestSWEbenchClientEvaluate:
    """Tests for SWEbenchClient evaluate methods."""

    @pytest.fixture
    def client(self):
        return SWEbenchClient()

    @pytest.fixture
    def sample_instance(self):
        return SWEbenchInstance(
            instance_id="django-django-1",
            repo="django/django",
            base_commit="abc123",
            problem_id="1",
            hints_text="",
            test_patch="test",
            code_language="python",
            version="1.0",
            problem_statement="Fix bug",
        )

    def test_evaluate_instance_basic_fallback(self, client, sample_instance):
        """Test evaluate_instance works with basic fallback."""
        patch_text = "diff --git a/test.py\n+++ b/test.py"

        result = client.evaluate_instance(sample_instance, patch_text, apply_patch=True)

        assert result.instance_id == "django-django-1"
        assert result.repo_name == "django/django"
        assert "swebench library not installed" in str(result.metadata)


class TestSWEbenchClientGeneratePatch:
    """Tests for SWEbenchClient generate_patch method."""

    @pytest.fixture
    def client(self):
        return SWEbenchClient()

    @pytest.fixture
    def sample_instance(self):
        return SWEbenchInstance(
            instance_id="django-django-1",
            repo="django/django",
            base_commit="abc123",
            problem_id="1",
            hints_text="Test hint",
            test_patch="test",
            code_language="python",
            version="1.0",
            problem_statement="Fix the bug",
        )

    def test_build_prompt_contains_key_fields(self, client, sample_instance):
        """Test that prompt includes all necessary fields."""
        prompt = client._build_swebench_prompt(sample_instance)

        assert "django/django" in prompt
        assert "django-django-1" in prompt
        assert "abc123" in prompt
        assert "python" in prompt
        assert "Fix the bug" in prompt
        assert "Test hint" in prompt
        assert "Generate the patch:" in prompt


class TestSWEbenchIntegration:
    """Integration tests for SWE-bench workflow."""

    def test_full_workflow(self):
        """Test the complete SWE-bench workflow."""
        # Setup
        dm = DataManager()
        config = BenchmarkConfig()
        client = SWEbenchClient(dm, config)

        # Create a mock instance
        instance = SWEbenchInstance(
            instance_id="test-repo-test-1",
            repo="test/repo",
            base_commit="abc123",
            problem_id="1",
            hints_text="Test hints",
            test_patch="test patch",
            code_language="python",
            version="1.0",
            problem_statement="Test problem",
        )

        # Generate patch (mocked)
        with patch.object(client, "generate_patch", return_value="diff --git a/test.py"):
            generated = client.generate_patch(instance, "/dev/null/fake-model")
            assert "diff --git" in generated

        # Evaluate (mocked)
        with patch.object(client, "evaluate_instance") as mock_eval:
            mock_eval.return_value = SWEbenchResult(
                repo_name="test/repo",
                instance_id="test-repo-test-1",
                resolved=True,
                test_results={"score": 1.0},
                patch_applied=True,
                patch_success=True,
            )
            result = client.evaluate_instance(instance, "diff --git a/test.py")
            assert result.resolved is True

        # Compute stats
        results = [
            SWEbenchResult(
                repo_name="test/repo",
                instance_id="test-repo-test-1",
                resolved=True,
            ),
            SWEbenchResult(
                repo_name="test/repo",
                instance_id="test-repo-test-2",
                resolved=False,
            ),
        ]
        stats = client.compute_stats(results)
        assert stats.total_instances == 2
        assert stats.resolved == 1
        assert stats.avg_test_score == pytest.approx(50.0, abs=0.1)
