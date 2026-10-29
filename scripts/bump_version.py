#!/usr/bin/env python3
"""Automatically determine version bump based on PR title and update pyproject.toml."""

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


def get_local_version():
    """Read current version from pyproject.toml."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()
    match = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', content)
    return match.group(1) if match else None


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

    # 4. Get latest version from PyPI (source of truth)
    pypi_version = get_latest_pypi_version()

    if not pypi_version:
        # No PyPI data available — trust local pyproject.toml
        local_version = get_local_version()
        if not local_version:
            print("Error: Could not parse version from pyproject.toml.")
            sys.exit(1)
        current_version_str = local_version
        print(f"Using local version (no PyPI data): {current_version_str}")
        # Bump from local version
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', local_version)
        major, minor, patch = map(int, match.groups())
        if bump_type == "minor":
            new_version = f"{major}.{minor + 1}.0"
        else:
            new_version = f"{major}.{minor}.{patch + 1}"
        print(f"Bumping version: {local_version} -> {new_version}")
        update_version_file(new_version)
        commit_and_push(new_version)
        print(f"Committed and pushed version bump to {new_version}")
        return

    # PyPI data available — compare local pyproject.toml with PyPI
    local_version = get_local_version()
    if not local_version:
        print("Error: Could not parse version from pyproject.toml.")
        sys.exit(1)

    current_version_str = local_version
    print(f"Local pyproject.toml version: {current_version_str}")
    print(f"Latest PyPI version: {pypi_version}")

    # Only bump if local version matches PyPI (meaning no bump happened yet for this commit)
    if local_version != pypi_version:
        print(f"Local ({local_version}) != PyPI ({pypi_version}). Bump already happened or out of sync. Skipping.")
        return

    # Calculate new version from PyPI
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', pypi_version)
    major, minor, patch = map(int, match.groups())

    if bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    else:
        new_version = f"{major}.{minor}.{patch + 1}"

    print(f"Bumping version: {pypi_version} -> {new_version}")

    # Update pyproject.toml and commit
    update_version_file(new_version)
    commit_and_push(new_version)
    print(f"Committed and pushed version bump to {new_version}")


if __name__ == "__main__":
    main()
