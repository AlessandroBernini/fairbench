"""Tests for fairbench.prompts — SWE-bench prompt engineering."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fairbench.prompts.swebench import (
    SWEbenchPromptBuilder,
    build_swebench_prompt,
    detect_architecture_from_filename,
    detect_architecture_from_gguf,
    detect_model_architecture,
    extract_patch,
)


class TestSWEbenchPromptBuilder:
    """Test the SWEbenchPromptBuilder class."""

    def test_build_qwen_template(self):
        """Test building a Qwen-optimized prompt."""
        builder = SWEbenchPromptBuilder()

        instance = MagicMock()
        instance.repo = "test/repo"
        instance.instance_id = "test-issue-123"
        instance.problem_statement = "Fix the bug"
        instance.hints_text = "Hint text"
        instance.test_patch = "diff test"

        prompt = builder.build(instance, "qwen")
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "Repository: test/repo" in prompt
        assert "Fix the bug" in prompt

    def test_build_llama_template(self):
        """Test building a Llama-optimized prompt."""
        builder = SWEbenchPromptBuilder()

        instance = MagicMock()
        instance.repo = "test/repo"
        instance.instance_id = "test-issue-123"
        instance.problem_statement = "Fix the bug"
        instance.hints_text = "Hint text"
        instance.test_patch = "diff test"

        prompt = builder.build(instance, "llama")
        assert "[INST]" in prompt
        assert "Repository: test/repo" in prompt

    def test_build_default_template(self):
        """Test building a default template."""
        builder = SWEbenchPromptBuilder()

        instance = MagicMock()
        instance.repo = "test/repo"
        instance.instance_id = "test-issue-123"
        instance.problem_statement = "Fix the bug"
        instance.hints_text = ""
        instance.test_patch = None

        prompt = builder.build(instance, "unknown_arch")
        assert "Repository: test/repo" in prompt
        assert "Problem Statement:" in prompt
        assert "Hints:" in prompt

    def test_default_params_qwen(self):
        """Test default params for Qwen models."""
        builder = SWEbenchPromptBuilder()
        params = builder.get_default_params("qwen")
        assert params["temperature"] == 0.1
        assert params["top_p"] == 0.9

    def test_default_params_llama(self):
        """Test default params for Llama models."""
        builder = SWEbenchPromptBuilder()
        params = builder.get_default_params("llama")
        assert params["temperature"] == 0.1

    def test_default_params_gpt(self):
        """Test default params for GPT models."""
        builder = SWEbenchPromptBuilder()
        params = builder.get_default_params("gpt")
        assert params["temperature"] == 0.15

    def test_default_params_unknown(self):
        """Test default params for unknown architecture."""
        builder = SWEbenchPromptBuilder()
        params = builder.get_default_params("unknown")
        assert params["temperature"] == 0.2


class TestGGUFArchitectureDetection:
    """Test GGUF header parsing for architecture detection."""

    def test_detect_from_gguf_qwen(self):
        """Test detecting Qwen architecture from GGUF header."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            # Write GGUF v2 magic + kv count + key-value pair
            f.write(b"GGUF")  # Magic
            f.write(b"\x02\x00\x00\x00")  # Version 2
            f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")  # kv_count = 1
            key = b"general.architecture"
            f.write(len(key).to_bytes(8, "little"))
            f.write(key)
            f.write(b"\x00\x00\x00\x00")  # value type: string (0)
            val = b"qwen2"
            f.write(len(val).to_bytes(8, "little"))
            f.write(val)
            temp_path = f.name

        try:
            arch = detect_architecture_from_gguf(temp_path)
            assert arch == "qwen"  # Normalized from "qwen2"
        finally:
            Path(temp_path).unlink()

    def test_detect_from_gguf_llama(self):
        """Test detecting Llama architecture from GGUF header."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF")
            f.write(b"\x02\x00\x00\x00")
            f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")
            key = b"general.architecture"
            f.write(len(key).to_bytes(8, "little"))
            f.write(key)
            f.write(b"\x00\x00\x00\x00")
            val = b"llama"
            f.write(len(val).to_bytes(8, "little"))
            f.write(val)
            temp_path = f.name

        try:
            arch = detect_architecture_from_gguf(temp_path)
            assert arch == "llama"
        finally:
            Path(temp_path).unlink()

    def test_detect_invalid_magic(self):
        """Test detection fails on invalid GGUF."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"NOTG")  # Invalid magic
            temp_path = f.name

        try:
            assert detect_architecture_from_gguf(temp_path) is None
        finally:
            Path(temp_path).unlink()


class TestFilenameArchitectureDetection:
    """Test filename-based architecture heuristics."""

    def test_detect_qwen_filename(self):
        """Test Qwen detection from filename."""
        assert detect_architecture_from_filename("qwen2.5-7b-q4.gguf") == "qwen"
        assert detect_architecture_from_filename("/path/to/qwen2-14b.gguf") == "qwen"

    def test_detect_llama_filename(self):
        """Test Llama detection from filename."""
        assert detect_architecture_from_filename("llama3-8b-instruct-q4.gguf") == "llama"
        assert detect_architecture_from_filename("llama3.1-70b.gguf") == "llama"

    def test_detect_mistral_filename(self):
        """Test Mistral detection from filename."""
        assert detect_architecture_from_filename("mistral-7b-v0.1.Q4_K_M.gguf") == "mistral"

    def test_detect_gemma_filename(self):
        """Test Gemma detection from filename."""
        assert detect_architecture_from_filename("gemma2-9b-q4.gguf") == "gemma"

    def test_detect_unknown_returns_none(self):
        """Test unknown architecture returns None."""
        assert detect_architecture_from_filename("unknown-model-v1.gguf") is None


class TestModelArchitectureDetection:
    """Test the combined architecture detection strategy."""

    def test_gguf_header_takes_precedence(self):
        """Test GGUF header parsing is used first."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF")
            f.write(b"\x02\x00\x00\x00")
            f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")
            key = b"general.architecture"
            f.write(len(key).to_bytes(8, "little"))
            f.write(key)
            f.write(b"\x00\x00\x00\x00")
            val = b"qwen2"
            f.write(len(val).to_bytes(8, "little"))
            f.write(val)
            temp_path = f.name

        try:
            arch = detect_model_architecture(temp_path)
            assert arch == "qwen"  # Normalized from "qwen2"
        finally:
            Path(temp_path).unlink()

    def test_filename_fallback(self):
        """Test filename detection as fallback."""
        with tempfile.NamedTemporaryFile(suffix="-qwen2.5-7b.gguf", delete=False) as f:
            f.write(b"GGUF")  # Valid magic but no proper header
            f.write(b"\x01\x00\x00\x00")  # Version 1 (no kv_pairs)
            temp_path = f.name

        try:
            arch = detect_model_architecture(temp_path)
            assert arch == "qwen"  # From filename heuristic
        finally:
            Path(temp_path).unlink()

    def test_default_fallback(self):
        """Test default architecture for invalid files."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"NOTG")  # Invalid magic
            temp_path = f.name

        try:
            arch = detect_model_architecture(temp_path)
            assert arch == "default"
        finally:
            Path(temp_path).unlink()


class TestPatchExtraction:
    """Test the robust patch extraction function."""

    def test_extract_from_diff_fence(self):
        """Test extraction from ```diff fenced block."""
        text = """Here is the patch:

```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
+new line
 existing
```

Done!"""
        result = extract_patch(text)
        assert "diff --git" in result
        assert "+new line" in result

    def test_extract_from_patch_fence(self):
        """Test extraction from ```patch fenced block."""
        text = "```patch\ndiff --git a/x.py b/x.py\n+fix\n```"
        result = extract_patch(text)
        assert "diff --git" in result

    def test_extract_from_generic_fence_with_diff(self):
        """Test extraction from generic code block containing diff markers."""
        text = "```\ndiff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n```"
        result = extract_patch(text)
        assert "diff --git" in result

    def test_extract_raw_diff(self):
        """Test extraction of raw diff (no code fences)."""
        text = "Here is the answer:\n\ndiff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        result = extract_patch(text)
        assert "diff --git" in result
        assert "-old" in result

    def test_extract_returns_text_when_no_patch(self):
        """Test that raw text is returned when no patch found."""
        text = "I think the answer is just to replace the variable."
        result = extract_patch(text)
        assert result == text.strip()


class TestBuildSWEbenchPrompt:
    """Test the main entry point build_swebench_prompt()."""

    def test_build_with_model_path(self):
        """Test building prompt with model path for detection."""
        instance = MagicMock()
        instance.repo = "test/repo"
        instance.instance_id = "issue-1"
        instance.problem_statement = "Fix bug"
        instance.hints_text = "Hint"
        instance.test_patch = "diff test"

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF")
            f.write(b"\x02\x00\x00\x00")
            f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")
            key = b"general.architecture"
            f.write(len(key).to_bytes(8, "little"))
            f.write(key)
            f.write(b"\x00\x00\x00\x00")
            val = b"qwen2"
            f.write(len(val).to_bytes(8, "little"))
            f.write(val)
            temp_path = f.name

        try:
            prompt, params = build_swebench_prompt(instance, temp_path)
            assert "<|system|>" in prompt  # Qwen template
            assert params["temperature"] == 0.1
        finally:
            Path(temp_path).unlink()

    def test_build_without_model_path(self):
        """Test building prompt without model path."""
        instance = MagicMock()
        instance.repo = "test/repo"
        instance.instance_id = "issue-1"
        instance.problem_statement = "Fix bug"
        instance.hints_text = ""
        instance.test_patch = None

        prompt, params = build_swebench_prompt(instance)
        assert "Problem Statement:" in prompt  # Default template
        assert params["temperature"] == 0.2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
