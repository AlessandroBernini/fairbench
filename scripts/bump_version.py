#!/usr/bin/env python3
"""Automatically determine version bump based on PR title and update pyproject.toml."""

import os
import re
import subprocess
import sys
from pathlib import Path


def get_latest_pypi_version(package_name="fairbench-ai"):
    """Get the latest version from PyPI."""
    try:
        result = subprocess.run(
            ["pip", "index", "versions", package_name],
            capture_output=True, text=True, timeout=10
        )
        # Output format: "Available versions: 0.5.0, 0.4.1, 0.4.0, ..."
        match = result.stdout.strip()
        if not match or "No versions found" in match:
            return None
        # Parse the first (latest) version
        versions = [v.strip() for v in match.split(":")[1].split(",")]
        return versions[0] if versions else None
    except Exception:
        return None


def determine_bump(commit_msg):
    """Return 'minor' for features, 'patch' otherwise."""
    t = commit_msg.lower()

    # Check for major features (e.g., "feat:", "feature", "breaking")
    if any(kw in t for kw in ["feat:", "feature:", "feature ", "breaking", "major"]):
        return "minor"  # Bump minor version

    # Default to patch bump (fix, chore, docs, etc.)
    return "patch"


def update_version_file(new_version):
    """Update pyproject.toml with the new version."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    # Regex to find version = "X.Y.Z"
    pattern = r'(version\s*=\s*")(\d+\.\d+\.\d+)"'
    replacement = f'\\g<1>{new_version}"'

    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        print("Error: No version pattern found in pyproject.toml")
        sys.exit(1)

    pyproject.write_text(new_content)


def commit_and_push(version):
    """Commit the version bump with [skip ci] to prevent recursion."""
    subprocess.run(["git", "add", "pyproject.toml"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"ci: bump version to {version} [skip ci]"],
        check=True
    )
    subprocess.run(["git", "push"], check=True)


def main():
    # 1. Check if we already bumped (look for [skip ci] in latest commit)
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True
    )
    if "[skip ci]" in result.stdout:
        print("Version already bumped in this run. Skipping.")
        return

    # 2. Get commit message from HEAD (already set by GitHub Actions)
    head_msg = result.stdout.strip()
    commit_msg = head_msg.split('\n')[0]  # First line is the subject
    print(f"Triggering commit message: '{commit_msg}'")

    # 3. Determine bump type from commit message
    bump_type = determine_bump(commit_msg)
    print(f"Determined bump type: {bump_type}")

    # 4. Get latest version from PyPI (source of truth), fallback to local pyproject.toml
    pypi_version = get_latest_pypi_version()
    current_version_str: str | None = None

    if pypi_version:
        current_version_str = pypi_version
        print(f"Using PyPI version as source of truth: {current_version_str}")
    else:
        # Fallback to local pyproject.toml
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        match = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', content)
        if not match:
            print("Error: Could not parse current version from pyproject.toml or PyPI.")
            sys.exit(1)
        current_version_str = match.group(1)
        print(f"Using local version as fallback: {current_version_str}")

    if current_version_str is None:
        print("Error: No valid version found.")
        sys.exit(1)

    # 5. Parse and calculate new version
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', current_version_str)
    major, minor, patch = map(int, match.groups())

    if bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    else:
        new_version = f"{major}.{minor}.{patch + 1}"

    print(f"Bumping version: {current_version_str} -> {new_version}")

    # 6. Check if bump is needed (avoid duplicate bumps)
    if pypi_version and current_version_str == pypi_version:
        print(f"Local version matches PyPI, bumping to {new_version}")
        update_version_file(new_version)
        commit_and_push(new_version)
    elif not pypi_version:
        # No PyPI data available, trust local state
        update_version_file(new_version)
        commit_and_push(new_version)
    else:
        print(f"PyPI version ({pypi_version}) differs from local. Skipping to avoid conflicts.")
        return

    print(f"Committed and pushed version bump to {new_version}")


if __name__ == "__main__":
    main()
