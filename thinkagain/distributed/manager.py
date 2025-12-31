"""Replica registry management for distributed execution."""

from __future__ import annotations

import threading
from typing import Callable

from .replica import ReplicaSpec


class ReplicaManager:
    """Manage replica registrations and lifecycle operations."""

    def __init__(self) -> None:
        self._registry: dict[str, ReplicaSpec] = {}
        self._lock = threading.RLock()

    def _list_specs(self) -> list[ReplicaSpec]:
        """Snapshot replica specs under lock."""
        with self._lock:
            return list(self._registry.values())

    def _resolve_name(self, name: str) -> str | None:
        if name in self._registry:
            return name
        if "." in name:
            return None
        matches = [key for key, spec in self._registry.items() if spec.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Replica name '{name}' is ambiguous. "
                f"Use a fully-qualified name instead."
            )
        return None

    def register(self, cls: type, n: int = 1) -> ReplicaSpec:
        """Register a replica class with desired instance count."""
        spec = ReplicaSpec(cls=cls, n=n)
        with self._lock:
            self._registry[spec.full_name] = spec
        return spec

    def replica(self, _cls: type | None = None, *, n: int = 1):
        """Decorator to register a class as a replica."""

        def decorator(cls: type) -> ReplicaSpec:
            return self.register(cls, n=n)

        if _cls is None:
            return decorator

        return decorator(_cls)

    def get_spec(self, name: str) -> ReplicaSpec | None:
        """Get replica spec by class name (simple or fully-qualified).

        Raises ValueError if a simple name is ambiguous.
        """
        with self._lock:
            key = self._resolve_name(name)
            return self._registry.get(key) if key is not None else None

    def get_all(self) -> dict[str, ReplicaSpec]:
        """Get all registered replicas keyed by fully qualified name."""
        with self._lock:
            return dict(self._registry)

    def clear(self) -> None:
        """Clear the replica registry (thread-safe). Useful for testing."""
        with self._lock:
            self._registry.clear()

    async def deploy_all(self) -> None:
        """Deploy all registered replicas with default constructor (async)."""
        for spec in self._list_specs():
            await spec.deploy()

    async def shutdown_all(self) -> None:
        """Shutdown all replica instances (async)."""
        for spec in self._list_specs():
            await spec.shutdown()


def replica(
    _cls: type | None = None,
    *,
    n: int = 1,
    manager: ReplicaManager | None = None,
) -> Callable[[type], ReplicaSpec] | ReplicaSpec:
    """Decorator that registers replicas with a manager.

    Defaults to the global manager when none is provided.
    """
    if manager is None:
        manager = get_default_manager()

    return manager.replica(_cls, n=n)


_default_manager = ReplicaManager()


def get_default_manager() -> ReplicaManager:
    """Return the global default replica manager."""
    return _default_manager


def set_default_manager(manager: ReplicaManager) -> None:
    """Replace the global default replica manager (useful for tests)."""
    global _default_manager
    _default_manager = manager
