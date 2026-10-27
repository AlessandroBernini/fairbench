#!/usr/bin/env python3
"""Custom installation for FairBench - handles llama-cpp-python C compilation."""

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstall(install):
    """Install core deps, then llama-cpp-python with wheel preference."""

    def run(self):
        install.run(self)

        # Skip llama-cpp-python in CI
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            print("\n⚠ Skipping llama-cpp-python installation in CI environment.")
            return

        # Attempt to install llama-cpp-python with wheel preference
        print("\nInstalling llama-cpp-python (LLM inference backend)...")
        env = os.environ.copy()
        env["LLAMA_CPP_FORCE_BUILD"] = "0"  # Prefer precompiled wheels

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                env=env,
                check=True,
            )
            print("✓ llama-cpp-python installed successfully.")
        except subprocess.CalledProcessError:
            print("\n⚠ Warning: Failed to install llama-cpp-python automatically.")
            print("This requires C build tools (cmake, build-essential).")
            print("\nTry running manually:")
            print("  pip install llama-cpp-python")
            print("\nIf that fails, install build dependencies first:")
            print("  sudo apt-get install cmake build-essential  # Debian/Ubuntu")
            print("  brew install cmake llvm                     # macOS")
            print("  sudo dnf install cmake gcc-c++              # Fedora/RHEL")
            sys.exit(1)


setup(
    name="fairbench-ai",
    version="0.3.0",
    description="FairBench - AI model benchmarking suite with SWE-bench integration",
    author="Alessandro Bernini",
    url="https://github.com/AlessandroBernini/fairbench",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1",
        "pyyaml>=6.0",
        "requests>=2.31",
        "psutil>=5.9",
        "lm-eval>=0.4",
        "swebench>=2.0",
        "datasets>=2.14",
        "rich>=13.0",
    ],
    extras_require={
        "ml": ["llama-cpp-python>=0.3.0"],
        "dev": ["pytest>=7.0", "ruff>=0.1", "mypy>=1.0", "huggingface-hub>=0.20"],
    },
    cmdclass={
        "install": CustomInstall,
    },
)
