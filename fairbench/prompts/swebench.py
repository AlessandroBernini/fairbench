"""SWE-bench prompt building and patch extraction utilities."""

from __future__ import annotations

import re
import struct
from typing import Any, ClassVar


class SWEbenchPromptBuilder:
    """Builds optimized prompts for SWE-bench patch generation."""

    TEMPLATES: ClassVar[dict[str, str]] = {
        "qwen": (
            "<|system|>\nYou are a software engineer. Fix the GitHub issue below by generating a unified diff patch.\n<|end|>\n"
            "<|user|>\n"
            "Repository: {repo}\nIssue: {instance_id}\n\n"
            "Problem:\n{problem_statement}\n\n"
            "Hints:\n{hints_text}\n\n"
            "Test file changes:\n```\n{test_patch}\n```\n\n"
            "Requirements:\n"
            "1. Output ONLY the unified diff patch\n"
            "2. Use 'diff --git' format with proper headers\n"
            "3. Be minimal — only change what's necessary\n"
            "4. Include all file paths in @@ hunks\n<|end|>\n"
            "<|assistant|>\n"
        ),
        "llama": (
            "[INST] You are a software engineer. Fix the GitHub issue below by generating a unified diff patch.\n\n"
            "Repository: {repo}\nIssue: {instance_id}\n\n"
            "Problem:\n{problem_statement}\n\n"
            "Hints:\n{hints_text}\n\n"
            "Test file changes:\n```\n{test_patch}\n```\n\n"
            "Requirements:\n"
            "1. Output ONLY the unified diff patch\n"
            "2. Use 'diff --git' format with proper headers\n"
            "3. Be minimal — only change what's necessary [/INST]"
        ),
        "gpt": (
            "You are a software engineer. Fix this GitHub issue by generating a unified diff patch.\n\n"
            "Repository: {repo}\nIssue ID: {instance_id}\n\n"
            "Problem Statement:\n{problem_statement}\n\n"
            "Hints:\n{hints_text}\n\n"
            "Test Patch:\n```\n{test_patch}\n```\n\n"
            "Requirements:\n"
            "1. Output ONLY the unified diff patch\n"
            "2. Use 'diff --git' format with proper headers\n"
            "3. Be minimal — only change what's necessary\n"
            "4. Include all file paths in @@ hunks"
        ),
        "default": (
            "You are a software engineer. Fix this GitHub issue by generating a unified diff patch.\n\n"
            "Repository: {repo}\nIssue ID: {instance_id}\n\n"
            "Problem Statement:\n{problem_statement}\n\n"
            "Hints:\n{hints_text}\n\n"
            "Test Patch:\n```\n{test_patch}\n```\n\n"
            "Requirements:\n"
            "1. Output ONLY the unified diff patch\n"
            "2. Use 'diff --git' format with proper headers\n"
            "3. Be minimal — only change what's necessary"
        ),
    }

    def build(
        self,
        instance: Any,
        model_architecture: str,
    ) -> str:
        """Build a prompt optimized for the given model architecture."""
        template = self.TEMPLATES.get(model_architecture, self.TEMPLATES["default"])
        return template.format(
            repo=instance.repo or "unknown",
            instance_id=getattr(instance, "instance_id", "unknown"),
            problem_statement=getattr(instance, "problem_statement", "N/A") or "N/A",
            hints_text=getattr(instance, "hints_text", "N/A") or "N/A",
            test_patch=getattr(instance, "test_patch", "") or "N/A",
        )

    def get_default_params(
        self, model_architecture: str
    ) -> dict[str, Any]:
        """Return default generation parameters optimized for the architecture."""
        if model_architecture in ("qwen", "llama"):
            return {"temperature": 0.1, "top_p": 0.9, "max_tokens": 4096}
        if model_architecture == "gpt":
            return {"temperature": 0.15, "top_p": 0.95, "max_tokens": 4096}
        # Default fallback
        return {"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096}


def detect_architecture_from_gguf(model_path: str) -> str | None:
    """Detect model architecture from GGUF header.

    Reads the 'general.architecture' key from the GGUF binary format.
    Returns None if detection fails (will fall back to default template).
    """
    try:
        with open(model_path, "rb") as f:
            # GGUF magic: 'GGUF'
            magic = f.read(4)
            if magic != b"GGUF":
                return None

            # Version 2+ has key-value pairs after metadata
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                return None
            version = struct.unpack("<I", version_bytes)[0]

            if version >= 2:
                # Read kv_count (uint64)
                kv_count_bytes = f.read(8)
                if len(kv_count_bytes) < 8:
                    return None
                kv_count = struct.unpack("<Q", kv_count_bytes)[0]

                # Scan for 'general.architecture' key
                for _ in range(kv_count):
                    # Key length (uint64) + key bytes
                    key_len_bytes = f.read(8)
                    if len(key_len_bytes) < 8:
                        return None
                    key_len = struct.unpack("<Q", key_len_bytes)[0]

                    key_data = f.read(key_len)
                    if key_data == b"general.architecture":
                        # Value type (uint32) + value data
                        val_type = struct.unpack("<I", f.read(4))[0]
                        if val_type == 0:  # string
                            str_len_bytes = f.read(8)
                            if len(str_len_bytes) < 8:
                                return None
                            str_len = struct.unpack("<Q", str_len_bytes)[0]
                            architecture = f.read(str_len).decode("utf-8")
                            return _normalize_architecture(architecture.lower())
                        # Skip other value types (not needed for arch)
                        return None
                    else:
                        # Skip value data — approximate
                        continue

            return None
    except (OSError, UnicodeDecodeError, struct.error):
        return None


# Architecture name normalization map
_ARCH_NORM = {
    "qwen2": "qwen",
    "qwen2.5": "qwen",
    "qwen3": "qwen",
    "llama": "llama",
    "llama3": "llama",
    "llama3.1": "llama",
    "gemma": "gemma",
    "gemma2": "gemma",
}


def _normalize_architecture(arch: str) -> str:
    """Normalize architecture name to template key."""
    for pattern, normalized in _ARCH_NORM.items():
        if arch.startswith(pattern):
            return normalized
    return arch


def detect_architecture_from_filename(model_path: str) -> str | None:
    """Detect architecture from model filename heuristics.

    Examples:
        - qwen2.5-7b-q4_k_m.gguf → 'qwen'
        - llama3-8b-instruct-q4.gguf → 'llama'
        - mistral-7b-v0.1.Q4_K_M.gguf → 'mistral'
    """
    import re as _re

    basename = model_path.lower()
    # Remove path components
    filename = _re.search(r"([\w.-]+)\.gguf$", basename)
    if not filename:
        return None

    name = filename.group(1)
    arch_map = {
        "qwen": "qwen",
        "llama": "llama",
        "llama3": "llama",
        "mistral": "mistral",
        "gemma": "gemma",
        "phi": "phi",
        "codestral": "codestral",
    }
    for pattern, arch in arch_map.items():
        if pattern in name:
            return arch
    return None


def detect_model_architecture(model_path: str) -> str:
    """Detect model architecture by trying multiple methods.

    Returns a string suitable for template lookup ('qwen', 'llama', 'gpt', or 'default').
    """
    # Method 1: GGUF header parsing (most reliable)
    arch = detect_architecture_from_gguf(model_path)
    if arch:
        return arch

    # Method 2: Filename heuristics
    arch = detect_architecture_from_filename(model_path)
    if arch:
        return arch

    # Method 3: Default fallback
    return "default"


def extract_patch(text: str) -> str:
    """Extract a unified diff patch from model output.

    Uses multiple strategies in order of reliability:
    1. Fenced code block with 'diff' or 'patch' language tag
    2. Any fenced code block containing 'diff --git'
    3. Raw 'diff --git' lines to end of text
    4. Return stripped text as-is

    Args:
        text: Model output string that may contain a patch.

    Returns:
        The extracted patch text, or the original text if no patch found.
    """
    # Strategy 1: Look for fenced code blocks with 'diff' or 'patch' lang tag
    match = re.search(r"```(?:diff|patch)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if result:
            return result

    # Strategy 2: Look for any fenced code block containing 'diff --git'
    match = re.search(r"```\s*\n(diff\s--git\s.*?)\n```", text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if result:
            return result

    # Strategy 3: Extract lines starting with 'diff --git' to end of relevant section
    if "diff --git" in text:
        lines = text.split("\n")
        diff_lines: list[str] = []
        in_diff = False
        for line in lines:
            if line.startswith("diff --git"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
        result = "\n".join(diff_lines).strip()
        if result:
            return result

    # Strategy 4: Return text as-is (model may have output raw diff)
    return text.strip()


def build_swebench_prompt(
    instance: Any,
    model_path: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build an optimized SWE-bench prompt and return it with default params.

    This is the main entry point used by the SWE-bench client.

    Args:
        instance: SWEbenchInstance with repo, instance_id, problem_statement, etc.
        model_path: Optional path to GGUF file for architecture detection.

    Returns:
        Tuple of (prompt_string, params_dict)
    """
    builder = SWEbenchPromptBuilder()

    # Detect architecture
    if model_path:
        arch = detect_model_architecture(model_path)
    else:
        arch = "default"

    prompt = builder.build(instance, arch)
    params = builder.get_default_params(arch)

    return prompt, params
