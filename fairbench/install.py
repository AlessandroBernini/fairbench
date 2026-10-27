"""Post-installation utilities for FairBench."""

from pathlib import Path


def print_post_install_message():
    """Print a friendly message after pip install."""
    print("\n" + "=" * 60)
    print("  FairBench installed successfully!")
    print("=" * 60)
    print()
    print("  📁 Config & Results directory:")
    print(f"     {Path.home() / '.fairbench'}")
    print()
    print("  ▶️  Next steps:")
    print("     1. Run 'fairbench init' to set up your config")
    print("     2. Run 'fairbench run <model.gguf>' to benchmark")
    print()
    print("  💡 Tip: Install all components (LM evaluation + SWE-bench):")
    print("     pip install fairbench-ai[all]")
    print()
    print("=" * 60)


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
        deps["llama-cpp-python"] = "✗ Not installed (needed for GGUF models)"

    # Check swebench
    try:
        import swebench  # noqa: F401
        deps["swebench"] = "✓ Installed"
    except ImportError:
        deps["swebench"] = "✗ Not installed (needed for SWE-bench)"

    return deps


def print_init_guide():
    """Print the init guide."""
    print("\n" + "=" * 60)
    print("  FairBench Setup Guide")
    print("=" * 60)
    print()
    
    # Show config path
    config_dir = Path.home() / ".fairbench"
    config_file = config_dir / "config.yaml"
    
    print(f"  📁 Config directory: {config_dir}")
    if config_file.exists():
        print(f"     ✓ Config file exists: {config_file}")
    else:
        print("     ✗ Config file NOT found")
        print("     → FairBench will create it automatically on first run")
    
    print()
    
    # Show results path
    results_dir = config_dir / "results"
    print(f"  📁 Results directory: {results_dir}")
    if not results_dir.exists():
        print("     → Will be created on first benchmark run")
    
    print()
    
    # Show dependencies
    print("  🔧 Dependencies status:")
    deps = check_dependencies()
    for name, status in deps.items():
        print(f"     {status} {name}")
    
    missing = [k for k, v in deps.items() if "✗" in v]
    if missing:
        print()
        print("  ⚠️  Missing dependencies:")
        for name in missing:
            print(f"     - {name}")
        
        install_cmd = f"pip install fairbench-ai[{','.join(missing)}]"
        print(f"\n  💡 Install with:")
        print(f"     {install_cmd}")
    
    # Show quick start
    print()
    print("  🚀 Quick Start:")
    print("     1. Place your GGUF model in ~/models/")
    print("     2. Run: fairbench run ~/models/your-model.gguf")
    print()
    print("=" * 60)


def cmd_init():
    """Interactive init command."""
    import click
    
    config_dir = Path.home() / ".fairbench"
    
    # Create directory structure
    print(f"📁 Creating FairBench directories...")
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
