"""Local model catalog — discover, index, and manage GGUF models."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

# GGUF magic bytes
GGUF_MAGIC = b"GGUF"

# Default scan directories
DEFAULT_SCAN_DIRS: list[Path] = [
    Path.home() / "models",
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".local" / "share" / "huggingface" / "hub",
    Path("/models"),
]

# Model index path
INDEX_PATH = Path.home() / ".fairbench" / "models_index.json"


class ModelInfo:
    """Metadata for a single GGUF model."""

    def __init__(
        self,
        path: Path,
        name: str = "",
        architecture: str = "",
        context_length: int = 0,
        quantization_version: int = 0,
        file_size_bytes: int = 0,
        quantization: str = "",
        **kwargs: Any,
    ) -> None:
        self.path = path
        self.name = name
        self.architecture = architecture
        self.context_length = context_length
        self.quantization_version = quantization_version
        self.file_size_bytes = file_size_bytes
        self.quantization = quantization
        self._extra = kwargs

    @property
    def file_size_gb(self) -> float:
        return self.file_size_bytes / (1024**3) if self.file_size_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "quantization_version": self.quantization_version,
            "file_size_bytes": self.file_size_bytes,
            "quantization": self.quantization,
            **self._extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        path = data.pop("path", "")
        return cls(path=Path(path), **data)

    def __repr__(self) -> str:
        return f"ModelInfo(name={self.name!r}, path={self.path})"


def _parse_gguf_header(path: Path) -> dict[str, Any]:
    """Parse GGUF binary header to extract model metadata."""
    info: dict[str, Any] = {
        "name": "",
        "architecture": "",
        "context_length": 0,
        "quantization_version": 0,
        "file_size_bytes": path.stat().st_size,
        "quantization": "",
    }

    try:
        with open(path, "rb") as f:
            # Check magic bytes
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                return info

            # Read version (uint64)
            version_data = f.read(8)
            if len(version_data) < 8:
                return info
            version = struct.unpack("<Q", version_data)[0]

            # Read tensor count (uint64)
            tensor_count_data = f.read(8)
            if len(tensor_count_data) < 8:
                return info
            struct.unpack("<Q", tensor_count_data)[0]

            # Read KV count (uint64)
            kv_count_data = f.read(8)
            if len(kv_count_data) < 8:
                return info
            kv_count = struct.unpack("<Q", kv_count_data)[0]

            # Read key-value pairs
            for _ in range(kv_count):
                # Key length + key bytes
                # GGUF v3 uses uint64 for key length, v2 uses uint32
                if version >= 3:
                    key_len_data = f.read(8)
                    if len(key_len_data) < 8:
                        break
                    key_len = struct.unpack("<Q", key_len_data)[0]
                else:
                    key_len_data = f.read(4)
                    if len(key_len_data) < 4:
                        break
                    key_len = struct.unpack("<I", key_len_data)[0]
                key_bytes = f.read(key_len)
                if len(key_bytes) < key_len:
                    break
                key = key_bytes.decode("utf-8", errors="ignore")

                # Value type (uint32)
                val_type_data = f.read(4)
                if len(val_type_data) < 4:
                    break
                val_type = struct.unpack("<I", val_type_data)[0]

                # Read value based on type
                value = _read_gguf_value(f, val_type)

                # Map known keys to info
                if key == "general.name":
                    info["name"] = str(value) if value else ""
                elif key == "general.architecture":
                    info["architecture"] = str(value) if value else ""
                elif key == "tokenizer.ggml.context_length":
                    info["context_length"] = int(value) if value else 0
                elif key == "general.quantization_version":
                    info["quantization_version"] = int(value) if value else 0
                elif key.startswith("general.quantization"):
                    info["quantization"] = str(value) if value else info.get("quantization", "")

    except (OSError, struct.error, UnicodeDecodeError):
        pass

    return info


def _read_gguf_value(f: Any, val_type: int) -> Any:
    """Read a GGUF value based on its type byte."""
    # GGUF value types (simplified)
    # 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
    # 6=uint64, 7=int64, 8=float32, 9=bool, 10=string, 11=array, 12=float16
    try:
        if val_type in (0, 1):
            return struct.unpack("<b", f.read(1))[0]
        elif val_type in (2, 3):
            return struct.unpack("<h", f.read(2))[0]
        elif val_type in (4, 5):
            return struct.unpack("<i", f.read(4))[0]
        elif val_type in (6, 7):
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == 8:
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == 9:
            return struct.unpack("<?", f.read(1))[0]
        elif val_type == 10:
            str_len_data = f.read(8)
            if len(str_len_data) < 8:
                return ""
            str_len = struct.unpack("<Q", str_len_data)[0]
            return f.read(str_len).decode("utf-8", errors="ignore")
        elif val_type == 12:
            return struct.unpack("<e", f.read(2))[0]
        else:
            # Unknown type — skip
            return None
    except (struct.error, OSError):
        return None


class ModelCatalog:
    """Local GGUF model catalog with scanning, caching, and search."""

    def __init__(
        self,
        scan_dirs: list[Path] | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.scan_dirs = scan_dirs or DEFAULT_SCAN_DIRS
        self.index_path = index_path or INDEX_PATH
        self._models: dict[str, ModelInfo] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cached index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                for model_data in data:
                    info = ModelInfo.from_dict(model_data)
                    # Verify file still exists
                    if info.path.exists():
                        self._models[info.name] = info
            except (json.JSONDecodeError, OSError):
                self._models = {}

    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._models.values()]
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def scan(self) -> dict[str, ModelInfo]:
        """Scan known directories for GGUF files and update index."""
        new_models: dict[str, ModelInfo] = {}

        for scan_dir in self.scan_dirs:
            if not scan_dir.exists():
                continue
            for gguf_file in sorted(scan_dir.rglob("*.gguf")):
                header = _parse_gguf_header(gguf_file)
                if not header["name"]:
                    continue

                name = header["name"]
                info = ModelInfo(path=gguf_file, **header)

                # Prefer existing cached info if file unchanged
                if name in self._models and self._models[name].path == gguf_file:
                    new_models[name] = self._models[name]
                else:
                    new_models[name] = info

        self._models = new_models
        self._save_index()
        return dict(new_models)

    def add_model(self, path: Path, metadata: dict[str, Any] | None = None) -> ModelInfo:
        """Add a model to the catalog (file must exist)."""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        header = metadata or _parse_gguf_header(path)
        name = header.get("name", path.stem)
        info = ModelInfo(path=path, **header)
        self._models[name] = info
        self._save_index()
        return info

    def remove_model(self, name: str) -> bool:
        """Remove a model from the catalog (keeps the file)."""
        if name in self._models:
            del self._models[name]
            self._save_index()
            return True
        return False

    def search(self, pattern: str) -> list[ModelInfo]:
        """Search models by name pattern (case-insensitive)."""
        pattern_lower = pattern.lower()
        return [
            m for m in self._models.values()
            if pattern_lower in m.name.lower() or pattern_lower in str(m.path).lower()
        ]

    def get_model(self, name_or_path: str) -> ModelInfo | None:
        """Get model by name or path."""
        # Check by name
        if name_or_path in self._models:
            return self._models[name_or_path]

        # Check by path
        for info in self._models.values():
            if str(info.path) == name_or_path or info.path.name == name_or_path:
                return info

        return None

    def list_models(self) -> list[ModelInfo]:
        """List all models in the catalog."""
        return sorted(self._models.values(), key=lambda m: m.name)

    def get_recommended_quant(self, model: ModelInfo) -> str:
        """Get recommended quantization based on model architecture."""
        arch = model.architecture.lower()
        if "llama" in arch or "llama" in arch:
            return "Q4_K_M"
        if "qwen" in arch or "qwen2" in arch:
            return "Q4_K_M"
        if "mistral" in arch:
            return "Q4_K_M"
        if "gemma" in arch:
            return "Q4_K_M"
        return "Q4_K_M"  # Default
