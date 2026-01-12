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
    from ..core.execution.executor import ExecutionContext
    from ..core.execution.replica import ReplicaHandle
    from ..core.tracing.context import TraceContext


# ---------------------------------------------------------------------------
# Service Call Executor (Plugin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceCallExecutor:
    """Executor for service calls.

    This executor is defined in the distributed module, not in core,
    demonstrating that new node types can be added without modifying
    the core execution infrastructure.

    The handle is passed as the first argument, and all replicas are
    invoked via their __call__ method.
    """

    async def execute(self, args: tuple, kwargs: dict, ctx: ExecutionContext) -> Any:
        """Execute a service call.

        Args:
            args: (handle, *call_args) - handle is first arg, rest are call args
            kwargs: Keyword arguments for the call
            ctx: Execution context

        Returns:
            Result from the service
        """
        # Get the service provider from context
        if ctx.service_provider is None:
            handle = args[0] if args else None
            service_desc = f"{handle.replica_class_name}" if handle else "service"
            raise RuntimeError(
                f"Service call to {service_desc} "
                f"requires service provider. Execute pipelines with services "
                f"inside 'with mesh:' block."
            )

        # Extract handle from first arg, rest are call args
        handle = args[0]
        call_args = args[1:]

        # Delegate to service provider (decoupled from mesh)
        return await ctx.service_provider.execute_service_call(
            handle, call_args, kwargs
        )

    def display_name(self) -> str:
        return "service"


# ---------------------------------------------------------------------------
# Service Tracing Hook
# ---------------------------------------------------------------------------


class ServiceTracingHook:
    """Hook that integrates service calls into graph tracing.

    This class implements the TracingHook protocol and is registered globally
    when a mesh context is active, allowing service calls to be properly
    recorded in the computation graph.
    """

    def __init__(self, trace_ctx: TraceContext):
        """Initialize with a trace context.

        Args:
            trace_ctx: Active tracing context to record service calls into
        """
        self.trace_ctx = trace_ctx

    async def record_call(self, args: tuple, kwargs: dict, handle) -> Any:
        """Record a service call during tracing.

        Args:
            args: Positional arguments for __call__
            kwargs: Keyword arguments for __call__
            handle: Handle being called (ReplicaHandle or TracedValue)

        Returns:
            TracedValue representing the future result of this call
        """
        from ..core.execution.replica import ReplicaHandle
        from ..core.graph.graph import InputRef, TracedValue

        if isinstance(handle, TracedValue):
            handle_ref = handle
        elif isinstance(handle, ReplicaHandle):
            handle_ref = InputRef(self.trace_ctx.get_resource_index(handle))
        else:
            from ..core.errors import TracingError

            raise TracingError("Service call requires a ReplicaHandle or traced input.")

        executor = ServiceCallExecutor()

        full_args = (handle_ref,) + args

        node_id = self.trace_ctx.add_node(executor, full_args, kwargs)
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
    execution of service calls by routing them to deployed service
    instances on the mesh.
    """

    def __init__(self, mesh: Any):
        """Initialize with a mesh instance.

        Args:
            mesh: Mesh instance that manages service deployments
        """
        self.mesh = mesh

    async def execute_service_call(
        self, handle: ReplicaHandle, args: tuple, kwargs: dict
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

        # Import here to avoid circular dependency at module level

        # Call the replica's execute method for consistent interface
        return await replica.execute(*args, **kwargs)


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
