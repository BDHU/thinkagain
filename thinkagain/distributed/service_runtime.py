"""Service execution runtime - bridges services to distributed mesh.

This module demonstrates the plugin architecture: it defines its own executor
(ServiceCallExecutor) and registers tracing hooks, without requiring any
changes to the core graph.py, executor.py, or tracing.py files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.tracing import register_tracing_plugin

if TYPE_CHECKING:
    from ..core.executor import ExecutionContext
    from ..core.replica import ReplicaHandle
    from ..core.tracing import TraceContext


# ---------------------------------------------------------------------------
# Service Call Executor (Plugin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceCallExecutor:
    """Executor for service method calls.

    This executor is defined in the distributed module, not in core,
    demonstrating that new node types can be added without modifying
    the core execution infrastructure.
    """

    handle_input_index: int
    method_name: str

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        """Execute a service method call."""
        # Get the service provider from context
        if ctx.service_provider is None:
            handle = ctx.inputs[self.handle_input_index]
            raise RuntimeError(
                f"Service call to {handle.replica_class_name}.{self.method_name} "
                f"requires service provider. Execute pipelines with services "
                f"inside 'with mesh:' block."
            )

        # Get the replica handle from inputs
        handle = ctx.inputs[self.handle_input_index]

        # Delegate to service provider (decoupled from mesh)
        return await ctx.service_provider.execute_service_call(
            handle, self.method_name, args, kwargs
        )

    def display_name(self) -> str:
        return f"service.{self.method_name}"


# ---------------------------------------------------------------------------
# Service Tracing Hook
# ---------------------------------------------------------------------------


class ServiceTracingHook:
    """Hook that integrates service calls into graph tracing.

    This class implements the TracingHook protocol and is registered globally
    when a mesh context is active, allowing service method calls to be properly
    recorded in the computation graph.
    """

    def __init__(self, trace_ctx: TraceContext):
        """Initialize with a trace context.

        Args:
            trace_ctx: Active tracing context to record service calls into
        """
        self.trace_ctx = trace_ctx

    async def record_call(
        self, method_name: str, args: tuple, kwargs: dict, handle: ReplicaHandle
    ) -> Any:
        """Record a service method call during tracing.

        Args:
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments
            handle: Replica handle being called

        Returns:
            TracedValue representing the future result of this call
        """
        # Get or register the handle's input index
        handle_idx = self.get_resource_index(handle)

        # Create executor for this service call
        executor = ServiceCallExecutor(
            handle_input_index=handle_idx,
            method_name=method_name,
        )

        # Record the node in the graph using the universal add_node
        node_id = self.trace_ctx.add_node(executor, args, kwargs)

        # Import here to avoid circular dependency
        from ..core.graph import TracedValue

        return TracedValue(node_id, self.trace_ctx)

    def get_resource_index(self, handle: ReplicaHandle) -> int:
        """Get or register a replica handle in the trace context.

        Args:
            handle: Replica handle to register

        Returns:
            Index where handle is stored in graph inputs
        """
        return self.trace_ctx.get_resource_index(handle)


# ---------------------------------------------------------------------------
# Service Provider (Execution)
# ---------------------------------------------------------------------------


class ServiceExecutionProvider:
    """Provider that executes service calls via mesh infrastructure.

    This class implements the ServiceProvider protocol and handles the actual
    execution of service method calls by routing them to deployed service
    instances on the mesh.
    """

    def __init__(self, mesh: Any):
        """Initialize with a mesh instance.

        Args:
            mesh: Mesh instance that manages service deployments
        """
        self.mesh = mesh

    async def execute_service_call(
        self, handle: ReplicaHandle, method_name: str, args: tuple, kwargs: dict
    ) -> Any:
        """Execute a service method call.

        Args:
            handle: Replica handle identifying the service
            method_name: Name of method to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result from the service method

        Raises:
            RuntimeError: If called outside mesh context
        """
        # Ensure service is deployed (auto-deploy if needed)
        await self.mesh._ensure_deployed(handle)

        # Get a replica instance
        replica = self.mesh.get_service_replica(handle)

        # Import here to avoid circular dependency at module level
        from ..core.runtime import maybe_await

        # Call the method
        method = getattr(replica, method_name)
        return await maybe_await(method, *args, **kwargs)


# ---------------------------------------------------------------------------
# Plugin Registration
# ---------------------------------------------------------------------------


# Self-register ServiceTracingHook as a tracing plugin when this module is imported
def _service_hook_factory(ctx: TraceContext) -> ServiceTracingHook | None:
    """Factory that creates ServiceTracingHook only if mesh is active.

    Args:
        ctx: TraceContext to attach the hook to

    Returns:
        ServiceTracingHook if mesh is active, None otherwise
    """
    from ..distributed import get_current_mesh

    mesh = get_current_mesh()
    if mesh:
        return ServiceTracingHook(ctx)
    return None


# Register the plugin factory on module import
# The factory will be called during tracing to check if a mesh is active
register_tracing_plugin(_service_hook_factory)
