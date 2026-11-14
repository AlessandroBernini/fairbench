"""Plugin registry and base class for benchmark plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from fairbench.config import CONFIG_PATH


class BenchmarkPlugin(ABC):
    """Abstract base class for all benchmark plugins."""

    name: str
    version: str
    description: str
    category: str  # "quality", "speed", "memory", "custom"

    def __init__(self) -> None:
        if not hasattr(self, 'name') or not hasattr(self, 'version'):
            raise ValueError("Plugin must define 'name' and 'version' class attributes")
        if not hasattr(self, 'description'):
            self.description = "No description provided"
        if not hasattr(self, 'category'):
            self.category = "custom"

    @abstractmethod
    def run(self, model_path: str, config: dict) -> dict[str, Any]:
        """Run the benchmark. Returns dict of results."""
        ...  # pragma: no cover

    @abstractmethod
    def get_metadata(self) -> dict[str, str | bool]:
        """Return benchmark metadata for listing."""
        ...  # pragma: no cover

    def __repr__(self) -> str:
        return f"<BenchmarkPlugin {self.name} v{self.version}>"


class PluginRegistry:
    """Discovers and manages loaded plugins."""

    _plugins: ClassVar[dict[str, BenchmarkPlugin]] = {}

    @classmethod
    def register(cls, plugin: BenchmarkPlugin) -> None:
        """Register a plugin instance."""
        if not isinstance(plugin, BenchmarkPlugin):
            raise TypeError(f"Expected BenchmarkPlugin, got {type(plugin).__name__}")
        cls._plugins[plugin.name] = plugin

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a plugin by name."""
        cls._plugins.pop(name, None)

    @classmethod
    def list_plugins(cls) -> list[BenchmarkPlugin]:
        """List all registered plugins."""
        return sorted(cls._plugins.values(), key=lambda p: p.name)

    @classmethod
    def get_plugin(cls, name: str) -> BenchmarkPlugin | None:
        """Get a plugin by name."""
        return cls._plugins.get(name)

    @classmethod
    def discover_builtin_plugins(cls) -> None:
        """Discover and register all built-in plugins."""
        import fairbench.plugins.builtin  # noqa: F401

    @classmethod
    def discover_file_plugins(cls) -> list[str]:
        """Discover file-based plugins from ~/.fairbench/plugins/custom/."""
        custom_dir = CONFIG_PATH.parent / "plugins" / "custom"
        discovered: list[str] = []

        if not custom_dir.exists():
            return discovered

        for entry in sorted(custom_dir.iterdir()):
            if entry.suffix == '.py' and entry.name != '__init__.py':
                try:
                    module_name = entry.stem
                    spec = importlib.util.spec_from_file_location(
                        f"custom_plugin_{module_name}", entry
                    )
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find plugin class in module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (inspect.isclass(attr) and
                                issubclass(attr, BenchmarkPlugin) and
                                attr is not BenchmarkPlugin):
                            instance = attr()
                            cls.register(instance)
                            discovered.append(attr_name)
                except (ImportError, AttributeError) as e:
                    print(f"⚠️  Failed to load plugin from {entry.name}: {e}")

        return discovered


# Auto-register on import
PluginRegistry.discover_builtin_plugins()
# Try to discover file plugins (may fail if CONFIG_PATH doesn't exist yet)
try:
    PluginRegistry.discover_file_plugins()
except (ImportError, AttributeError):  # pragma: no cover - CONFIG_PATH may not exist at import time
    pass
