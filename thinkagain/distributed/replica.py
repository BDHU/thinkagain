"""Replica specification for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import get_backend as _get_backend

try:
    from .profiling import record_replica_call as _record_replica_call
except (ImportError, AttributeError):

    def _record_replica_call(_name: str) -> None:
        return None


@dataclass
class ReplicaSpec:
    """Specification for a replica class.

    Defines resource requirements per instance. The actual number of instances
    is determined at deployment time (either manually or via optimizer).
    """

    cls: type
    cpus: int = 0  # CPUs per instance (0 for GPU-only workers)
    gpus: int = 0  # GPUs per instance (0 for CPU-only workers)

    def __post_init__(self):
        """Validate resource specifications."""
        from .validation import validate_resources

        validate_resources(self.cpus, self.gpus)

    @property
    def name(self) -> str:
        return self.cls.__name__

    @property
    def full_name(self) -> str:
        return f"{self.cls.__module__}.{self.cls.__qualname__}"

    async def deploy(self, instances: int = 1, *args, **kwargs) -> None:
        """Deploy instances via the configured backend (async).

        Args:
            instances: Number of instances to deploy (default: 1)
            *args: Constructor arguments for replica instances
            **kwargs: Constructor keyword arguments for replica instances
        """
        await _get_backend().deploy(self, instances=instances, *args, **kwargs)

    async def shutdown(self) -> None:
        """Shutdown instances via the configured backend (async)."""
        await _get_backend().shutdown(self)

    def get(self) -> Any:
        """Get next instance via round-robin (sync).

        Raises:
            RuntimeError: If replica is not deployed.
        """
        backend = _get_backend()
        if not backend.is_deployed(self):
            raise RuntimeError(
                f"Replica '{self.name}' not deployed. "
                f"Call 'await {self.name}.deploy()' first."
            )

        instance = backend.get_instance(self)

        _record_replica_call(self.name)

        return instance
