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

    async def deploy(self, spec: "ReplicaSpec", *args, **kwargs) -> None: ...
    async def shutdown(self, spec: "ReplicaSpec") -> None: ...
    def get_instance(self, spec: "ReplicaSpec") -> Any: ...
    def is_deployed(self, spec: "ReplicaSpec") -> bool: ...
