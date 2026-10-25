#!/usr/bin/env python3
"""Publish FairBench to PyPI.

Usage:
    python scripts/publish.py [--dry-run]

This script:
1. Verifies pyproject.toml is correct
2. Checks for .pypirc credentials
3. Builds the package
4. Uploads to PyPI (or test PyPI with --dry-run)

Requires:
    pip install build twine
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check_requirements():
    """Check that all requirements are met."""
    errors = []

    # Check pyproject.toml exists
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        errors.append("pyproject.toml not found")
        return errors, None

    # Check .pypirc exists
    pypirc = ROOT / ".pypirc"
    if not pypirc.exists():
        pypirc_example = ROOT / ".env.pypirc.example"
        if pypirc_example.exists():
            errors.append(
                f".pypirc not found. Copy {pypirc_example} to .pypirc "
                "and fill in your credentials."
            )
        else:
            errors.append(".pypirc not found")
        return errors, None

    # Check build dependencies
    for pkg in ["build", "twine"]:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True,
        )
        if result.returncode != 0:
            errors.append(f"{pkg} not installed. Run: pip install {pkg}")

    return errors, pyproject


def build_package():
    """Build the package."""
    print("🔨 Building package...")
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        return False
    print("✅ Build complete")
    return True


def upload_package(dry_run=False):
    """Upload to PyPI."""
    dist_dir = ROOT / "dist"
    if not dist_dir.exists():
        print("❌ No dist/ directory. Run build first.")
        return False

    files = list(dist_dir.glob("*"))
    if not files:
        print("❌ No files in dist/")
        return False

    target = "testpypi" if dry_run else "pypi"
    print(f"📤 Uploading to {target}...")

    cmd = [
        sys.executable, "-m", "twine", "upload",
        *files,
        "--repository", target,
    ]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("FairBench PyPI Publisher")
    print("=" * 60)

    dry_run = "--dry-run" in sys.argv

    # Check requirements
    errors, pyproject = check_requirements()
    if errors:
        print("\n❌ Requirements not met:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"\n📦 pyproject.toml: {pyproject}")

    # Confirm
    target = "test PyPI (dry run)" if dry_run else "PyPI"
    print(f"\n🎯 Target: {target}")
    print(f"\n⚠️  This will upload packages to {target}.")

    if not dry_run:
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            sys.exit(0)

    # Build
    if not build_package():
        sys.exit(1)

    # Upload
    if upload_package(dry_run=dry_run):
        print("\n✅ Publish successful!")
    else:
        print("\n❌ Upload failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
