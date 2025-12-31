"""In-process backend for running replicas locally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec

from .utils import RoundRobinPool, PoolBackendMixin


class LocalBackend(PoolBackendMixin):
    """Local backend keeps replica instances within the same process (sync)."""

    def __init__(self):
        self._pool = RoundRobinPool()

    async def deploy(
        self, spec: "ReplicaSpec", instances: int = 1, *args, **kwargs
    ) -> None:
        """Deploy replica instances locally.

        Args:
            spec: ReplicaSpec with resource requirements
            instances: Number of instances to deploy
            *args: Constructor arguments
            **kwargs: Constructor keyword arguments
        """
        self._deploy_to_pool(spec.name, spec.cls, instances, args, kwargs)

    async def shutdown(self, spec: "ReplicaSpec") -> None:
        self._shutdown_from_pool(spec.name)

    def get_instance(self, spec: "ReplicaSpec") -> Any:
        """Get next instance in round-robin order."""
        return self._get_from_pool(spec.name)

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        return self._is_deployed_in_pool(spec.name)
