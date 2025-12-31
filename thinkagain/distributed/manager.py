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

    def register(self, cls: type, cpus: int = 0, gpus: int = 0) -> ReplicaSpec:
        """Register a replica class with resource requirements.

        Args:
            cls: The replica class to register
            cpus: CPUs per instance (0 for GPU-only workers)
            gpus: GPUs per instance (0 for CPU-only workers)

        At least one of cpus or gpus must be > 0.

        Returns:
            ReplicaSpec for the registered class
        """
        spec = ReplicaSpec(cls=cls, cpus=cpus, gpus=gpus)
        with self._lock:
            self._registry[spec.full_name] = spec
        return spec

    def replica(
        self,
        _cls: type | None = None,
        *,
        cpus: int | None = None,
        gpus: int | None = None,
    ):
        """Decorator to register a class as a replica.

        Args:
            cpus: CPUs per instance (None means 0, i.e., not used)
            gpus: GPUs per instance (None means 0, i.e., not used)

        At least one of cpus or gpus must be specified and > 0.

        Examples:
            @replica(cpus=2, gpus=1)  # Mixed: 2 CPUs + 1 GPU per instance
            class HybridModel:
                ...

            @replica(gpus=1)  # GPU-only: 1 GPU, 0 CPUs per instance
            class LLMPool:
                ...

            @replica(cpus=4)  # CPU-only: 4 CPUs, 0 GPUs per instance
            class VectorDB:
                ...
        """

        def decorator(cls: type) -> ReplicaSpec:
            final_cpus = 0 if cpus is None else cpus
            final_gpus = 0 if gpus is None else gpus
            return self.register(cls, cpus=final_cpus, gpus=final_gpus)

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
    cpus: int | None = None,
    gpus: int | None = None,
    manager: ReplicaManager | None = None,
) -> Callable[[type], ReplicaSpec] | ReplicaSpec:
    """Decorator that registers replicas with a manager.

    Args:
        cpus: CPUs per instance (None means 0)
        gpus: GPUs per instance (None means 0)
        manager: ReplicaManager to use (default: global manager)

    At least one of cpus or gpus must be specified and > 0.

    Examples:
        @replica(cpus=4, gpus=1)  # Mixed: 4 CPUs + 1 GPU per instance
        class HybridModel:
            ...

        @replica(gpus=2)  # GPU-only: 2 GPUs, 0 CPUs per instance
        class vLLM:
            ...

        @replica(cpus=8)  # CPU-only: 8 CPUs, 0 GPUs per instance
        class VectorDB:
            ...
    """
    if manager is None:
        manager = get_default_manager()

    return manager.replica(_cls, cpus=cpus, gpus=gpus)


_default_manager = ReplicaManager()


def get_default_manager() -> ReplicaManager:
    """Return the global default replica manager."""
    return _default_manager


def set_default_manager(manager: ReplicaManager) -> None:
    """Replace the global default replica manager (useful for tests)."""
    global _default_manager
    _default_manager = manager
