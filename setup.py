#!/usr/bin/env python3
"""Setup script for FairBench (pip install compatibility)."""

from setuptools import setup, find_packages

setup(
    name="fairbench",
    version="0.2.0",
    description="Fair benchmarking for local LLMs under 70B on consumer hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alessandro Bernini",
    author_email="alessandro@mair.it",
    url="https://github.com/AlessandroBernini/fairbench",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "click>=8.1.7",
        "requests>=2.31.0",
        "psutil>=5.9.0",
        "llama-cpp-python>=0.3.0",
        "lm-eval>=0.4.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "ruff>=0.1.0", "mypy>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "fairbench=fairbench.cli:cli",
        ],
    },
)
