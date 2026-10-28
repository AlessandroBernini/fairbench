#!/usr/bin/env python3
"""Automatically determine version bump based on PR title and update pyproject.toml."""

import re
import subprocess
import sys
from pathlib import Path


def determine_bump(commit_msg):
    """Return 'minor' for features, 'patch' otherwise."""
    # Normalize title to lowercase
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
    subprocess.run(["git", "commit", "-m", f"ci: bump version to {version} [skip ci]"], check=True)
    subprocess.run(["git", "push"], check=True)


def main():
    # 1. Check if we already bumped (look for [skip ci] in latest commit)
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True
    )
    if "[skip ci]" in result.stdout:
        print("Version already bumped in this run. Skipping.")
        return

    # 2. Get commit message from HEAD (already set to trigger commit by GitHub Actions)
    head_msg = result.stdout.strip()
    commit_msg = head_msg.split('\n')[0]  # First line is the subject
    print(f"Triggering commit message: '{commit_msg}'")

    # 3. Determine bump type from commit message
    bump_type = determine_bump(commit_msg)
    print(f"Determined bump type: {bump_type}")

    # 4. Calculate new version
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()
    match = re.search(r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', content)
    
    if not match:
        print("Error: Could not parse current version.")
        sys.exit(1)

    major, minor, patch = map(int, match.groups())
    
    if bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    else:
        new_version = f"{major}.{minor}.{patch + 1}"

    print(f"Bumping version: {match.group(1)}.{match.group(2)}.{match.group(3)} -> {new_version}")

    # 5. Update file and commit
    update_version_file(new_version)
    
    # Use GITHUB_TOKEN for push (standard in CI)
    subprocess.run(["git", "push"], check=True)
    print(f"Committed and pushed version bump to {new_version}")


if __name__ == "__main__":
    main()
