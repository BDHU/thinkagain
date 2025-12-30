"""Replica registry management for distributed execution."""

from __future__ import annotations

import threading
from typing import Any, Callable

from .replica import ReplicaSpec


class ReplicaHandle:
    """Public handle for interacting with a registered replica."""

    __slots__ = ("_spec",)

    def __init__(self, spec: ReplicaSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.cls.__name__

    @property
    def full_name(self) -> str:
        return f"{self._spec.cls.__module__}.{self._spec.cls.__qualname__}"

    @property
    def cls(self) -> type:
        return self._spec.cls

    @property
    def spec(self) -> ReplicaSpec:
        return self._spec

    def deploy(self, *args, **kwargs) -> None:
        """Deploy instances. Idempotent; call shutdown() to redeploy."""
        self._spec.deploy_instances(*args, **kwargs)

    def shutdown(self) -> None:
        """Shutdown all instances."""
        self._spec.shutdown_instances()

    def get(self) -> Any:
        """Get next instance via round-robin."""
        return self._spec.get_instance()


class ReplicaManager:
    """Manage replica registrations and lifecycle operations."""

    def __init__(self) -> None:
        self._registry: dict[str, ReplicaHandle] = {}
        self._lock = threading.RLock()

    def _list_specs(self) -> list[ReplicaSpec]:
        """Snapshot replica specs under lock."""
        with self._lock:
            return [handle.spec for handle in self._registry.values()]

    def _registry_key(self, cls: type) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    def _resolve_name(self, name: str) -> str | None:
        if name in self._registry:
            return name
        if "." in name:
            return None
        matches = [
            key for key, handle in self._registry.items() if handle.cls.__name__ == name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Replica name '{name}' is ambiguous. "
                f"Use a fully-qualified name instead."
            )
        return None

    def register(self, cls: type, n: int = 1) -> ReplicaHandle:
        """Register a replica class with desired instance count."""
        spec = ReplicaSpec(cls=cls, n=n)
        handle = ReplicaHandle(spec)
        with self._lock:
            self._registry[self._registry_key(cls)] = handle
        return handle

    def replica(self, _cls: type | None = None, *, n: int = 1):
        """Decorator to register a class as a replica."""

        def decorator(cls: type) -> ReplicaHandle:
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
            handle = self._registry.get(key) if key is not None else None
            return handle.spec if handle is not None else None

    def get_handle(self, name: str) -> ReplicaHandle | None:
        """Get replica handle by class name (simple or fully-qualified).

        Raises ValueError if a simple name is ambiguous.
        """
        with self._lock:
            key = self._resolve_name(name)
            return self._registry.get(key) if key is not None else None

    def get_all(self) -> dict[str, ReplicaHandle]:
        """Get all registered replicas keyed by fully qualified name."""
        with self._lock:
            return dict(self._registry)

    def clear(self) -> None:
        """Clear the replica registry (thread-safe). Useful for testing."""
        with self._lock:
            self._registry.clear()

    def deploy_all(self) -> None:
        """Deploy all registered replicas with default constructor."""
        for spec in self._list_specs():
            spec.deploy_instances()

    def shutdown_all(self) -> None:
        """Shutdown all replica instances."""
        for spec in self._list_specs():
            spec.shutdown_instances()


def replica(
    _cls: type | None = None,
    *,
    n: int = 1,
    manager: ReplicaManager | None = None,
) -> Callable[[type], ReplicaHandle] | ReplicaHandle:
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
