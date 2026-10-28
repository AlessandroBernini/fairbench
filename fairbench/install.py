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
        lines.append("  ⚠️  Config file missing (will be auto-created on first run)")
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
        _total, _used, free = shutil.disk_usage(str(Path.home()))
        free_gb = free / (1024**3)
        if free_gb >= 5:
            lines.append(f"  ✅ Disk space available: {free_gb:.1f} GB")
        else:
            lines.append(f"  ⚠️  Low disk space: {free_gb:.1f} GB (recommend at least 5 GB)")
            warnings.append("Free up space if you plan to run benchmarks")
    except OSError:
        # Could not determine disk usage (shouldn't happen on normal systems)
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
    import click

    from fairbench.config import load_config, save_config

    config_dir = Path.home() / ".fairbench"

    # Create directory structure
    print("📁 Creating FairBench directories...")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "results").mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {config_dir}")
    print()

    # Interactive configuration
    print("⚙️  Configuration Wizard")
    print("-" * 40)
    
    config_file = config_dir / "config.yaml"
    existing_config = load_config() if config_file.exists() else None
    
    # Ask for leaderboard username
    username = click.prompt(
        "👤 Leaderboard username",
        default=existing_config.api_url.split("/")[-1] if existing_config and "fairbench.vercel.app" in existing_config.api_url else "",
        show_default=True,
    )
    
    # Ask for publish setting
    publish = click.confirm(
        "📤 Enable leaderboard publishing?",
        default=True,
    )
    
    # Ask for GPU layers
    gpu_layers = click.prompt(
        "🎮 Default GPU layers",
        type=int,
        default=-1,
        show_default=True,
    )
    
    # Ask for quantization
    quant = click.prompt(
        "📦 Default quantization",
        default="q4_k_m",
        show_default=True,
    )
    
    print()

    # Create/update config
    if existing_config:
        print("✏️  Updating existing configuration...")
    else:
        print("📝 Creating new configuration...")
    
    config = load_config()
    config.api_url = f"https://fairbench.vercel.app/api/{username}" if username else config.api_url
    config.publish = publish
    config.default_n_gpu_layers = gpu_layers
    config.default_quant = quant
    save_config(config)
    print(f"   ✓ Config saved to {config_file}")
    
    # Show dependency status
    print()
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
    print("   1. Download a model: fairbench download Qwen/Qwen2.5-1.5B-Instruct")
    print("   2. Or place your GGUF model (e.g., ~/models/model.gguf)")
    print("   3. Run: fairbench run ~/models/model.gguf")
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


def cmd_download(model_id: str, filename: str | None = None, local_dir: str | None = None):
    """Download a GGUF model from HuggingFace.
    
    MODEL_ID is the HuggingFace repository ID (e.g., 'Qwen/Qwen2.5-1.5B-Instruct').
    FILENAME is optional specific file to download (defaults to first .gguf file found).
    LOCAL_DIR is where to save (defaults to ~/models/).
    """
    from pathlib import Path
    
    # Determine target directory
    target_dir = Path(local_dir) if local_dir else Path.home() / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading model: {model_id}")
    print(f"   Target directory: {target_dir}")
    print()
    
    # Try using huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        files = list_repo_files(model_id, repo_type="model")
        gguf_files = [f for f in files if f.endswith(".gguf") or "gguf" in f.lower()]
        
        if not gguf_files:
            print("  ❌ No GGUF files found in repository")
            print("  💡 Try a different model ID that has GGUF variants")
            return
        
        if filename:
            # Download specific file
            if filename not in gguf_files:
                print(f"  ⚠️  '{filename}' not found, available files:")
                for f in gguf_files[:10]:  # Show first 10
                    print(f"     - {f}")
                return
            
            local_file = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(target_dir),
                force_download=False,
            )
        else:
            # Download first GGUF file
            selected_file = gguf_files[0]
            print(f"   Auto-selecting: {selected_file}")
            local_file = hf_hub_download(
                repo_id=model_id,
                filename=selected_file,
                local_dir=str(target_dir),
                force_download=False,
            )
        
        print("\n  ✅ Model downloaded successfully!")
        print(f"   Saved to: {local_file}")
        print()
        print("  🚀 Run benchmark:")
        print(f"     fairbench run {local_file}")
        
    except ImportError:
        # Fallback to huggingface-cli
        print("  ℹ️  Using huggingface-cli (huggingface_hub not installed)")
        import subprocess
        
        cmd = ["huggingface-cli", "download", model_id, "--local-dir", str(target_dir)]
        
        if filename:
            cmd.extend(["--include", filename])
        else:
            # Try to download only GGUF files
            print("  ⚠️  No filename specified. Use --filename to select specific GGUF file.")
            print(f"     Example: fairbench download {model_id} --filename Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
            return
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=False)
            if result.returncode == 0:
                print(f"\n  ✅ Model downloaded successfully to {target_dir}")
            else:
                print("\n  ❌ Download failed. Make sure you have huggingface-cli installed:")
                print("     pip install huggingface_hub")
        except FileNotFoundError:
            print("\n  ❌ huggingface-cli not found.")
            print("  💡 Install with:")
            print("     pip install huggingface_hub")
            print()
            print("  Then run:")
            print(f"     huggingface-cli download {model_id} --local-dir {target_dir}")
