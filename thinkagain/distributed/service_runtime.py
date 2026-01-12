"""Service execution runtime - bridges services to the distributed mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.execution.replica import ReplicaHandle


class ServiceExecutionProvider:
    """Provider that executes service calls via mesh infrastructure.

    This class implements the ServiceProvider protocol and handles the actual
    execution of service calls by routing them to deployed service instances
    on the mesh.
    """

    def __init__(self, mesh: Any):
        """Initialize with a mesh instance.

        Args:
            mesh: Mesh instance that manages service deployments
        """
        self.mesh = mesh

    async def execute_service_call(
        self, handle: "ReplicaHandle", args: tuple, kwargs: dict
    ) -> Any:
        """Execute a service call.

        Args:
            handle: Replica handle identifying the service
            args: Positional arguments for __call__
            kwargs: Keyword arguments for __call__

        Returns:
            Result from the service

        Raises:
            RuntimeError: If called outside mesh context
        """
        # Ensure service is deployed (auto-deploy if needed)
        await self.mesh._ensure_deployed(handle)

        # Get a replica instance
        replica = self.mesh.get_service_replica(handle)

        # Call the replica's execute method for consistent interface
        return await replica.execute(*args, **kwargs)
