#!/usr/bin/env python3
"""Validate registry: check for outdated models.

Usage:
    python scripts/cron/validate_registry.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Import registry
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fairbench.registry import MODEL_REGISTRY, MAX_MODEL_AGE_DAYS


def validate():
    """Validate all models in registry."""
    outdated = []
    valid = []
    
    for name, info in MODEL_REGISTRY.items():
        if name == "custom":
            continue
            
        added = info.get("added")
        if not added:
            outdated.append((name, "no 'added' date"))
            continue
            
        try:
            added_dt = datetime.strptime(added, "%Y-%m-%d")
            added_dt = added_dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - added_dt).days
            
            if age > MAX_MODEL_AGE_DAYS:
                outdated.append((name, f"{age} days old (added {added})"))
            else:
                valid.append((name, f"{age} days old"))
        except ValueError:
            outdated.append((name, f"invalid date: {added}"))
    
    # Report
    print(f"📋 FairBench Registry Validation")
    print(f"   Max age: {MAX_MODEL_AGE_DAYS} days")
    print(f"   Valid: {len(valid)}")
    print(f"   Outdated: {len(outdated)}")
    print()
    
    if valid:
        print("✅ Valid models:")
        for name, age in valid:
            print(f"   • {name} ({age})")
        print()
    
    if outdated:
        print("⚠️  Outdated models (remove them):")
        for name, reason in outdated:
            print(f"   • {name} — {reason}")
        print()
    
    if outdated:
        print("🔧 To fix: remove outdated models from fairbench/registry.py")
        print("   or add them to the removal list in .github/old-models.md")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(validate())
