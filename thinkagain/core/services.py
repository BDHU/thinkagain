"""Service binding metadata for @node functions."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


def bind_service(**service_handles: Any):
    """Attach service handle metadata to a @node for tracing/caching."""
    if not service_handles:
        raise ValueError("bind_service() requires at least one service handle")

    def decorator(fn: Callable[..., Awaitable[Any]]):
        # Store bindings as metadata for runtime introspection
        setattr(fn, "_service_bindings", dict(service_handles))
        return fn

    return decorator
