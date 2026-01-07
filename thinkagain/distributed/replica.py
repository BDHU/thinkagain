"""Replica specification for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizer import ThroughputFunc

from .runtime import get_backend as _get_backend

try:
    from .profiling import record_replica_call as _record_replica_call
except (ImportError, AttributeError):

    def _record_replica_call(_name: str) -> None:
        return None


@dataclass
class ReplicaSpec:
    """Specification for a replica class.

    Defines resource requirements and scaling configuration per instance.
    The actual number of instances is determined at deployment time
    (either manually or via optimizer).
    """

    cls: type
    cpus: int = 0  # CPUs per instance (0 for GPU-only workers)
    gpus: int = 0  # GPUs per instance (0 for CPU-only workers)

    # Scaling configuration (optional - only needed for auto-scaling/optimization)
    throughput: ThroughputFunc | None = (
        None  # Throughput model for auto-scaling/optimization
    )

    # Deprecated: min/max instances moved to Constraints.per_replica_constraints
    # These are kept for backward compatibility
    min_instances: int = 1  # Minimum instances for auto-scaling
    max_instances: int = 100  # Maximum instances for auto-scaling
    max_utilization: float = 0.8  # Target max utilization for auto-scaling

    def __post_init__(self):
        """Validate resource specifications."""
        if self.cpus < 0 or self.gpus < 0:
            raise ValueError("Resource counts cannot be negative")
        if self.cpus == 0 and self.gpus == 0:
            raise ValueError(
                "At least one resource must be > 0. "
                "Specify cpus=N for CPU-only or gpus=N for GPU-only."
            )

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

    def to_replica_config(self):
        """Convert to ReplicaConfig for optimizer.

        Returns:
            ReplicaConfig for use with optimizer/auto-scaler

        Raises:
            ValueError: If throughput is not configured
        """
        from .optimizer import ReplicaConfig

        if self.throughput is None:
            raise ValueError(
                f"Replica '{self.name}' does not have throughput configured. "
                f"Add throughput parameter to @replica decorator."
            )

        return ReplicaConfig(
            name=self.name,
            throughput_func=self.throughput,
            cpus_per_instance=self.cpus,
            gpus_per_instance=self.gpus,
            min_instances=self.min_instances,
            max_instances=self.max_instances,
        )
