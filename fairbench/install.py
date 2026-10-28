"""Post-installation utilities for FairBench."""

import sys
from pathlib import Path


def print_post_install_message():
    """Print a friendly message after pip install."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  🎉 FairBench installed successfully!")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  ✅ Next steps to get started:")
    lines.append("")
    lines.append("  1️⃣  Run 'fairbench init' to set up your config")
    lines.append("  2️⃣  Place your GGUF model (e.g., ~/models/model.gguf)")
    lines.append("  3️⃣  Run 'fairbench run ~/models/model.gguf'")
    lines.append("")
    lines.append("  💡 Tip: Install all components (LM evaluation + SWE-bench):")
    lines.append("     pip install fairbench-ai[all]")
    lines.append("")
    lines.append("=" * 60)
    print("\n".join(lines))


def check_dependencies():
    """Check which dependencies are available."""
    deps = {}

    # Check lm_eval
    try:
        import lm_eval  # noqa: F401
        deps["lm-eval"] = "✓ Installed"
    except ImportError:
        deps["lm-eval"] = "✗ Not installed (needed for quality benchmarks)"

    # Check llama-cpp-python
    try:
        import llama_cpp  # noqa: F401
        deps["llama-cpp-python"] = "✓ Installed"
    except ImportError:
        deps["llama-cpp-python"] = "✗ Not installed (needed for speed/memory benchmarks)"

    # Check swebench
    try:
        import swebench  # noqa: F401
        deps["swebench"] = "✓ Installed"
    except ImportError:
        deps["swebench"] = "✗ Not installed (needed for SWE-bench)"

    return deps


def cmd_verify():
    """Verify FairBench installation and system readiness."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  🔍 FairBench System Check")
    lines.append("=" * 60)
    lines.append("")

    all_ok = True
    warnings = []

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if (sys.version_info.major, sys.version_info.minor) >= (3, 10):
        lines.append(f"  ✅ Python {py_version} (required: 3.10+)")
    else:
        lines.append(f"  ❌ Python {py_version} (requires 3.10+)")
        all_ok = False

    # Check config directory
    config_dir = Path.home() / ".fairbench"
    if config_dir.exists():
        lines.append(f"  ✅ Config directory exists: {config_dir}")
    else:
        lines.append(f"  ⚠️  Config directory missing (will be created on first run): {config_dir}")
        warnings.append("Run 'fairbench init' to create it manually")

    # Check config file
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        lines.append(f"  ✅ Config file found: {config_file}")
    else:
        lines.append(f"  ⚠️  Config file missing (will be auto-created on first run)")
        warnings.append("Run 'fairbench init' to create it manually")

    # Check dependencies
    lines.append("")
    lines.append("  🔧 Dependencies:")

    deps = check_dependencies()
    for name, status in deps.items():
        if "✓" in status:
            lines.append(f"  ✅ {name}")
        else:
            lines.append(f"  ⚠️  {name} — not installed")
            all_ok = False

    # Check disk space (rough estimate)
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(Path.home()))
        free_gb = free / (1024**3)
        if free_gb >= 5:
            lines.append(f"  ✅ Disk space available: {free_gb:.1f} GB")
        else:
            lines.append(f"  ⚠️  Low disk space: {free_gb:.1f} GB (recommend at least 5 GB)")
            warnings.append("Free up space if you plan to run benchmarks")
    except Exception:
        pass

    # Summary
    lines.append("")
    if all_ok:
        lines.append("  🎉 Your system is ready to run FairBench!")
        lines.append("")
        lines.append("  Quick start:")
        lines.append("     1. fairbench run ~/models/model.gguf")
        lines.append("     2. Or try a demo: fairbench quick-start")
    else:
        lines.append("  ⚠️  Some items need attention:")
        for w in warnings:
            lines.append(f"     • {w}")

    lines.append("")
    lines.append("=" * 60)
    print("\n".join(lines))


def cmd_quick_start():
    """Show what's available for a quick demo."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  🚀 FairBench Quick Start Demo")
    lines.append("=" * 60)
    lines.append("")

    # Check if a model is available
    model_path = Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    if not model_path.exists():
        lines.append("  ❌ No model found at ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        lines.append("")
        lines.append("  To run a demo, you need a GGUF model:")
        lines.append("")
        lines.append("  Option 1: Download a small model (recommended)")
        lines.append("     huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct")
        lines.append("       --include '*.gguf*' --local-dir ~/models/")
        lines.append("")
        lines.append("  Option 2: Use your own model")
        lines.append("     fairbench run /path/to/your/model.gguf")
        lines.append("")
        lines.append("  💡 Tip: Start with a small model (1.5B-3B) for fast testing")
        lines.append("")
        lines.append("=" * 60)
        print("\n".join(lines))
        return

    # Check if llama-cpp-python is available
    try:
        import llama_cpp  # noqa: F401
        has_llama = True
    except ImportError:
        has_llama = False

    if not has_llama:
        lines.append("  ❌ llama-cpp-python not installed")
        lines.append("")
        lines.append("  Speed/memory benchmarks require llama-cpp-python:")
        lines.append("     pip install llama-cpp-python")
        lines.append("")

    # Check dependencies
    deps = check_dependencies()
    has_lm_eval = "✓" in deps.get("lm-eval", "")
    has_swebench = "✓" in deps.get("swebench", "")

    if not has_lm_eval:
        lines.append("  ⚠️  lm-eval not installed — will skip quality benchmarks")
        lines.append("")

    # Run demo
    lines.append("  📋 Demo plan:")
    lines.append(f"     Model: {model_path}")
    if has_llama:
        lines.append("     ✅ Speed benchmark (TTFT, TPOT)")
        lines.append("     ✅ Memory benchmark (VRAM, RAM)")
    else:
        lines.append("     ❌ Speed/memory benchmarks require llama-cpp-python")

    if has_lm_eval:
        lines.append("     ✅ Quality benchmarks (lite mode)")
    else:
        lines.append("     ❌ Quality benchmarks require lm-eval")

    if has_swebench:
        lines.append("     ⏭️  SWE-bench (requires many instances — skipped in demo)")

    lines.append("")
    lines.append("  🎯 To run the full demo:")
    lines.append(f"     fairbench run {model_path} --category all --lite")
    lines.append("")
    lines.append("=" * 60)
    print("\n".join(lines))


def cmd_init():
    """Interactive setup wizard — creates config, checks dependencies, shows paths."""
    config_dir = Path.home() / ".fairbench"

    # Create directory structure
    print("📁 Creating FairBench directories...")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "results").mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {config_dir}")
    print()

    # Show current config or create default
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        print("📝 Creating default configuration...")
        from fairbench.config import load_config, save_config
        config = load_config()
        save_config(config)
        print(f"   ✓ Config saved to {config_file}")
    else:
        print(f"📋 Config file exists: {config_file}")
        with open(config_file) as f:
            import yaml
            config_data = yaml.safe_load(f) or {}
        if config_data:
            print("   Current settings:")
            for key, value in config_data.items():
                print(f"     - {key}: {value}")

    print()

    # Show dependency status
    print("🔧 Checking dependencies...")
    deps = check_dependencies()
    for name, status in deps.items():
        icon = "✅" if "✓" in status else "❌"
        print(f"   {icon} {name}: {status}")

    missing = [k for k, v in deps.items() if "✗" in v]
    if missing:
        print()
        print("  ⚠️  Missing dependencies will affect benchmark capabilities:")
        install_cmd = f"pip install fairbench-ai[{','.join(missing)}]"
        print(f"  Install with: {install_cmd}")

    # Show paths
    print()
    print("📁 FairBench directories:")
    print(f"   Config:     {config_file}")
    print(f"   Results:    {config_dir / 'results'}")
    print()

    # Quick start
    print("🚀 Quick start:")
    print("   1. Place your GGUF model (e.g., ~/models/model.gguf)")
    print("   2. Run: fairbench run ~/models/model.gguf")
    print()
    print("=" * 60)


def cmd_show_paths():
    """Show all FairBench paths."""
    config_dir = Path.home() / ".fairbench"

    print("\nFairBench Paths:")
    print("=" * 40)
    print(f"Config directory: {config_dir}")
    print(f"Config file:      {config_dir / 'config.yaml'}")
    print(f"Results dir:      {config_dir / 'results'}")
    print()

    # Show package location
    try:
        import fairbench
        print(f"Package location: {fairbench.__file__}")
    except ImportError:
        print("Package location: (fairbench not installed)")

    print("=" * 40 + "\n")
