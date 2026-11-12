"""Hardware profiling and caching for FairBench."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fairbench.config import CONFIG_PATH
from fairbench.hardware import detect_hardware


@dataclass
class HardwareSnapshot:
    """Snapshot of system hardware for profiling and caching."""

    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    gpu_available: bool
    gpu_type: str | None
    gpu_vram_gb: float
    os_platform: str
    os_version: str
    kernel_version: str
    profile_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d = asdict(self)
        d.pop("profile_hash", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareSnapshot:
        """Deserialize from dict."""
        d = dict(d)
        d.pop("profile_hash", None)
        instance = cls(**d)
        instance.profile_hash = _compute_hash(instance)
        return instance


_CACHE_FILE = CONFIG_PATH.parent / "hardware_profile.json"


def _compute_hash(snapshot: HardwareSnapshot) -> str:
    """Compute profile hash from snapshot fields (excluding profile_hash itself)."""
    data = snapshot.to_dict()
    return hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()


def _get_hardware_snapshot() -> HardwareSnapshot:
    """Create a HardwareSnapshot from current hardware detection."""
    hw = detect_hardware()
    snapshot = HardwareSnapshot(
        cpu_model=hw.cpu_model,
        cpu_cores=hw.cpu_cores,
        cpu_threads=hw.cpu_threads,
        total_ram_gb=hw.total_ram_gb,
        gpu_available=hw.gpu_available,
        gpu_type=hw.gpu_type or "",
        gpu_vram_gb=hw.gpu_vram_gb,
        os_platform=hw.os_platform,
        os_version=platform.release(),
        kernel_version=platform.version(),
    )
    snapshot.profile_hash = _compute_hash(snapshot)
    return snapshot


def get_cached_hardware_snapshot() -> HardwareSnapshot | None:
    """Load cached hardware snapshot if it exists and hardware hasn't changed.

    Returns None if no cache exists or hardware has changed.
    """
    if not _CACHE_FILE.exists():
        return None

    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    cached = HardwareSnapshot.from_dict(data)
    current = _get_hardware_snapshot()

    if cached.profile_hash == current.profile_hash:
        return cached

    return None


def save_hardware_snapshot(snapshot: HardwareSnapshot) -> Path:
    """Save hardware snapshot to cache file."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)
    return _CACHE_FILE


def clear_hardware_cache() -> bool:
    """Clear hardware profile cache. Returns True if file was removed."""
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
        return True
    return False


def get_hardware_profile() -> HardwareSnapshot:
    """Get current hardware snapshot, using cache if available."""
    cached = get_cached_hardware_snapshot()
    if cached:
        return cached

    snapshot = _get_hardware_snapshot()
    save_hardware_snapshot(snapshot)
    return snapshot


def get_hw_profile_info() -> dict[str, Any]:
    """Get hardware profile info for display."""
    profile = get_hardware_profile()
    return {
        "profile_hash": profile.profile_hash,
        "cpu_model": profile.cpu_model,
        "cpu_cores": profile.cpu_cores,
        "cpu_threads": profile.cpu_threads,
        "total_ram_gb": profile.total_ram_gb,
        "gpu_available": profile.gpu_available,
        "gpu_type": profile.gpu_type,
        "gpu_vram_gb": profile.gpu_vram_gb,
        "os_platform": profile.os_platform,
        "os_version": profile.os_version,
        "kernel_version": profile.kernel_version,
        "cached": _CACHE_FILE.exists(),
    }
