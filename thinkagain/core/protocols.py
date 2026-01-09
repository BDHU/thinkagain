"""Protocol definitions for extensible execution."""

from __future__ import annotations

from typing import Any, Protocol


class ServiceProvider(Protocol):
    """Provider interface for executing service calls.

    This protocol decouples the executor from the concrete service runtime,
    allowing different backends (local, distributed, remote) to provide
    service execution capabilities.
    """

    async def execute_service_call(
        self, handle: Any, method_name: str, args: tuple, kwargs: dict
    ) -> Any:
        """Execute a service method call.

        Args:
            handle: Service handle (opaque to executor)
            method_name: Name of method to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result of the service method call
        """
        ...


class TracingHook(Protocol):
    """Hook for recording operations during graph tracing.

    This protocol allows external systems (like service runtime) to participate
    in tracing without coupling the core tracing infrastructure to specific concepts.
    """

    async def record_call(
        self, method_name: str, args: tuple, kwargs: dict, context: Any
    ) -> Any:
        """Record a traced operation.

        Args:
            method_name: Name of the operation being traced
            args: Positional arguments
            kwargs: Keyword arguments
            context: Additional context (e.g., service handle)

        Returns:
            Traced value representing the operation's result
        """
        ...

    def get_resource_index(self, resource: Any) -> int:
        """Get or register a resource (e.g., service handle) in the trace.

        Args:
            resource: Resource to register

        Returns:
            Index where resource is stored in graph inputs
        """
        ...
