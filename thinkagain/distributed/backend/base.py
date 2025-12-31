"""Backend protocol definitions for distributed replicas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec


class Backend(Protocol):
    """Protocol all replica backends must implement.

    deploy() and shutdown() are async (I/O operations).
    get_instance() and is_deployed() are sync (no I/O).
    """

    async def deploy(
        self, spec: "ReplicaSpec", instances: int = 1, *args, **kwargs
    ) -> None:
        """Deploy replica instances.

        Args:
            spec: ReplicaSpec with resource requirements (cpus, gpus)
            instances: Number of instances to deploy
            *args: Constructor arguments for instances
            **kwargs: Constructor keyword arguments for instances
        """
        ...

    async def shutdown(self, spec: "ReplicaSpec") -> None: ...
    def get_instance(self, spec: "ReplicaSpec") -> Any: ...
    def is_deployed(self, spec: "ReplicaSpec") -> bool: ...
