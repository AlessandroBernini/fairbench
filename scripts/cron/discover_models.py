#!/usr/bin/env python3
"""Weekly discovery script for new GGUF models on HuggingFace.

Usage:
    python scripts/cron/discover_models.py

Outputs a markdown file with proposed new models for review.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi, list_models, ModelSearchArguments
except ImportError:
    print("❌ Install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def discover_new_gguf_models(days_threshold: int = 30) -> list[dict]:
    """Discover new GGUF models from HuggingFace."""
    api = HfApi()
    new_models = []
    
    # Search for recent GGUF models
    # We search for models with "gguf" in their tags or filenames
    search_args = ModelSearchArguments()
    
    # Get recent models with GGUF tag
    models = list_models(
        sort="downloads",
        direction=-1,
        full=True,
    )
    
    seen_repos = set()
    
    for model in models:
        if len(new_models) >= 50:  # Limit results
            break
            
        repo_id = model.id
        if repo_id in seen_repos:
            continue
        seen_repos.add(repo_id)
        
        # Check if GGUF
        tags = getattr(model, 'tags', []) or []
        if 'gguf' not in [t.lower() for t in tags]:
            continue
            
        # Check creation date
        created = getattr(model, 'created_at', None)
        if not created:
            continue
            
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_dt).days
        except (ValueError, AttributeError):
            continue
            
        if age_days > days_threshold:
            continue
            
        # Check if it has quantized variants
        model_info = api.model_info(repo_id)
        siblings = getattr(model_info, 'siblings', None) or []
        gguf_files = [f.rfilename for f in siblings if f.rfilename.lower().endswith('.gguf')]
        
        if not gguf_files:
            continue
            
        # Determine best quantization (prefer Q4_K_M)
        quant = "q4_k_m"
        for f in gguf_files:
            if "q4_k_m" in f.lower():
                quant = "q4_k_m"
                break
            elif "q4_k" in f.lower():
                quant = "q4_k"
                break
                
        # Estimate parameters from repo name
        size_b = _estimate_size(repo_id)
        
        new_models.append({
            "repo": repo_id,
            "file_pattern": f"{repo_id.split('/')[-1]}-{quant}.gguf",
            "size_b": size_b,
            "context": _estimate_context(repo_id),
            "age_days": age_days,
            "downloads": getattr(model, 'downloads', 0) or 0,
        })
    
    return new_models


def _estimate_size(repo_id: str) -> float:
    """Estimate model size from repo name."""
    name = repo_id.lower().split('/')[-1]
    if '70b' in name:
        return 70.0
    elif '34b' in name:
        return 34.0
    elif '13b' in name or '14b' in name:
        return 14.0
    elif '8b' in name or '7b' in name:
        return 7.0
    elif '3b' in name:
        return 3.0
    elif '1b' in name:
        return 1.0
    else:
        return None  # Unknown


def _estimate_context(repo_id: str) -> int:
    """Estimate context window from repo name."""
    name = repo_id.lower()
    if '128k' in name or '131072' in name:
        return 131072
    elif '32k' in name or '32768' in name:
        return 32768
    elif '8k' in name or '8192' in name:
        return 8192
    elif '4k' in name or '4096' in name:
        return 4096
    else:
        return 4096  # Default


def generate_report(new_models: list[dict]) -> str:
    """Generate markdown report of new models."""
    lines = [
        "# New GGUF Models Discovered",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Models found:** {len(new_models)}",
        "",
        "## Proposed additions to registry",
        "",
        "| Name | Repo | Quant | Params | Context | Age | Downloads |",
        "|------|------|-------|--------|---------|-----|-----------|",
    ]
    
    for m in new_models:
        name = m['repo'].split('/')[-1]
        lines.append(
            f"| `{name}` | `{m['repo']}` | `{m['file_pattern']}` | "
            f"{m['size_b']}B | {m['context']} | {m['age_days']}d | {m['downloads']:,}"
        )
    
    lines.extend([
        "",
        "## Action needed",
        "",
        "Review these models and add valid ones to `fairbench/registry.py`.",
        "Run `fairbench validate-registry` to check for outdated models.",
    ])
    
    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent.parent.parent / "discovered"
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Discovering new GGUF models...")
    new_models = discover_new_gguf_models(days_threshold=30)
    
    report = generate_report(new_models)
    output_file = output_dir / f"new-models-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    output_file.write_text(report)
    
    print(f"✅ Found {len(new_models)} new models")
    print(f"📄 Report: {output_file}")
    
    # Also output to stdout for CI
    print("\n" + report)


if __name__ == "__main__":
    main()
