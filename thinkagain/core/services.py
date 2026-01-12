"""Service binding metadata for @node functions."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


def bind_service(**service_handles: Any):
    """Attach service handle metadata to a @node for tracing/caching."""
    if not service_handles:
        raise ValueError("bind_service() requires at least one service handle")

    def decorator(fn: Callable[..., Awaitable[Any]]):
        # Store bindings as metadata for runtime introspection
        bindings = dict(service_handles)
        setattr(fn, "_service_bindings", bindings)
        node_fn = getattr(fn, "_node_fn", None)
        if node_fn is not None:
            setattr(node_fn, "_service_bindings", bindings)
        return fn

    return decorator


def get_service_bindings(fn: Any) -> dict[str, Any] | None:
    """Return service bindings attached to a node, if any."""
    return getattr(fn, "_service_bindings", None)


def register_service_bindings(ctx: Any, fn: Any) -> None:
    """Register bound services in the trace context for cache/resource tracking."""
    bindings = get_service_bindings(fn)
    if not bindings:
        return
    for handle in bindings.values():
        ctx.get_resource_index(handle)
