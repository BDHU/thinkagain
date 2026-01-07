"""Replica registry management for distributed execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .optimizer import ThroughputFunc

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

    def register(
        self,
        cls: type,
        cpus: int = 0,
        gpus: int = 0,
        throughput: ThroughputFunc | None = None,
        min_instances: int = 1,
        max_instances: int = 100,
        max_utilization: float = 0.8,
    ) -> ReplicaSpec:
        """Register a replica class with resource requirements and scaling config.

        Args:
            cls: The replica class to register
            cpus: CPUs per instance (0 for GPU-only workers)
            gpus: GPUs per instance (0 for GPU-only workers)
            throughput: Throughput model for auto-scaling/optimization (optional)
            min_instances: Minimum instances for auto-scaling (default: 1, deprecated)
            max_instances: Maximum instances for auto-scaling (default: 100, deprecated)
            max_utilization: Target max utilization for auto-scaling (default: 0.8)

        At least one of cpus or gpus must be > 0.

        Returns:
            ReplicaSpec for the registered class
        """
        spec = ReplicaSpec(
            cls=cls,
            cpus=cpus,
            gpus=gpus,
            throughput=throughput,
            min_instances=min_instances,
            max_instances=max_instances,
            max_utilization=max_utilization,
        )
        with self._lock:
            self._registry[spec.full_name] = spec
        return spec

    def replica(
        self,
        _cls: type | None = None,
        *,
        cpus: int | None = None,
        gpus: int | None = None,
        throughput: ThroughputFunc | None = None,
        min_instances: int = 1,
        max_instances: int = 100,
        max_utilization: float = 0.8,
    ):
        """Decorator to register a class as a replica.

        Args:
            cpus: CPUs per instance (None means 0, i.e., not used)
            gpus: GPUs per instance (None means 0, i.e., not used)
            throughput: Throughput model for auto-scaling/optimization (optional)
            min_instances: Minimum instances for auto-scaling (default: 1, deprecated)
            max_instances: Maximum instances for auto-scaling (default: 100, deprecated)
            max_utilization: Target max utilization for auto-scaling (default: 0.8)

        At least one of cpus or gpus must be specified and > 0.

        Examples:
            # Simple deployment (no auto-scaling)
            @replica(cpus=2)
            class VectorDB:
                ...

            # With optimization
            @replica(cpus=2, throughput=linear_throughput(200.0))
            class VectorDB:
                ...

            # GPU-based with batching
            @replica(gpus=1, throughput=batched_throughput(50.0, 32))
            class LLM:
                ...
        """

        def decorator(cls: type) -> ReplicaSpec:
            final_cpus = 0 if cpus is None else cpus
            final_gpus = 0 if gpus is None else gpus
            return self.register(
                cls,
                cpus=final_cpus,
                gpus=final_gpus,
                throughput=throughput,
                min_instances=min_instances,
                max_instances=max_instances,
                max_utilization=max_utilization,
            )

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
    throughput: ThroughputFunc | None = None,
    min_instances: int = 1,
    max_instances: int = 100,
    max_utilization: float = 0.8,
) -> Callable[[type], ReplicaSpec] | ReplicaSpec:
    """Decorator that registers replicas with the global manager.

    Args:
        cpus: CPUs per instance (None means 0)
        gpus: GPUs per instance (None means 0)
        throughput: Throughput model for auto-scaling/optimization (optional)
        min_instances: Minimum instances for auto-scaling (default: 1, deprecated)
        max_instances: Maximum instances for auto-scaling (default: 100, deprecated)
        max_utilization: Target max utilization for auto-scaling (default: 0.8)

    At least one of cpus or gpus must be specified and > 0.

    Examples:
        # Simple deployment
        @replica(cpus=8)
        class VectorDB:
            ...

        # With optimization
        @replica(cpus=2, throughput=linear_throughput(200.0))
        class VectorDB:
            ...

        # GPU-based with batching
        @replica(gpus=1, throughput=batched_throughput(50.0, 32))
        class LLM:
            ...
    """
    manager = get_default_manager()

    return manager.replica(
        _cls,
        cpus=cpus,
        gpus=gpus,
        throughput=throughput,
        min_instances=min_instances,
        max_instances=max_instances,
        max_utilization=max_utilization,
    )


_default_manager = ReplicaManager()


def get_default_manager() -> ReplicaManager:
    """Return the global default replica manager."""
    return _default_manager


def set_default_manager(manager: ReplicaManager) -> None:
    """Replace the global default replica manager (useful for tests)."""
    global _default_manager
    _default_manager = manager
