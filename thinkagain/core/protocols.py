"""Protocol definitions for extensible execution."""

from __future__ import annotations

from typing import Any, Protocol


class ServiceProvider(Protocol):
    """Provider interface for executing service calls.

    This protocol decouples the executor from the concrete service runtime,
    allowing different backends (local, distributed, remote) to provide
    service execution capabilities.
    """

    async def execute_service_call(self, handle: Any, args: tuple, kwargs: dict) -> Any:
        """Execute a service method call.

        Args:
            handle: Service handle (opaque to executor)
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result of the service method call
        """
        ...
