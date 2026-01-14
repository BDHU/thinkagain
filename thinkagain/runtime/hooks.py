"""Execution hooks for pluggable extensions.

Provides a lightweight hook system for extending executor behavior without
tight coupling. Extensions (like distributed execution and actor execution)
can register hooks to intercept and modify execution flow.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Protocol

from .context import get_current_execution_context
from ..resources import get_current_mesh

__all__ = [
    "ExecutionHook",
    "register_hook",
    "unregister_hook",
    "get_hooks",
    "clear_hooks",
    "register_distributed_hooks",
    "unregister_distributed_hooks",
    "dynamic_service_hook",
]


class ExecutionHook(Protocol):
    """Protocol for execution hooks.

    Hooks can intercept op execution and either:
    1. Return (True, result) to short-circuit execution with a custom result
    2. Return (False, None) to continue with normal execution
    """

    async def __call__(
        self,
        op: Any,
        args: tuple,
        kwargs: dict,
    ) -> tuple[bool, Any]:
        """Intercept op execution.

        Args:
            op: Op or ServiceOp being executed
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            (handled, result) tuple:
            - If handled=True, result is used instead of calling op.fn
            - If handled=False, normal execution continues
        """
        ...


def register_hook(hook: ExecutionHook) -> None:
    """Register an execution hook.

    Hooks are checked in registration order. The first hook that returns
    (True, result) will short-circuit execution.

    Args:
        hook: Hook to register
    """
    hooks = get_current_execution_context().hooks
    if hook not in hooks:
        hooks.append(hook)


def unregister_hook(hook: ExecutionHook) -> None:
    """Unregister an execution hook.

    Args:
        hook: Hook to remove
    """
    hooks = get_current_execution_context().hooks
    if hook in hooks:
        hooks.remove(hook)


def get_hooks() -> list[ExecutionHook]:
    """Get all registered hooks (for testing/debugging)."""
    return get_current_execution_context().hooks.copy()


def clear_hooks() -> None:
    """Clear all hooks (for testing)."""
    get_current_execution_context().hooks.clear()


# --------------------------------------------------------------------------- #
# Distributed execution hook
# --------------------------------------------------------------------------- #


async def _execute_with_profile(name: str, call: Callable[[], Awaitable[Any]]) -> Any:
    """Execute with optional profiling."""
    from . import profiling

    if profiling.get_profiler() is None:
        return await call()

    start_time = time.perf_counter()
    try:
        return await call()
    finally:
        duration = time.perf_counter() - start_time
        profiling.record_service_call(name, duration=duration)


async def distributed_execution_hook(
    op: Any,
    args: tuple,
    kwargs: dict,
) -> tuple[bool, Any]:
    """Hook to handle distributed execution of service classes.

    Args:
        op: Op being executed
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        (handled, result) tuple:
        - If fn has _service_config and mesh is active: (True, result)
        - Otherwise: (False, None) to continue with normal execution
    """
    # Check if class is a service
    fn = getattr(op, "fn", None)
    if fn is None or not hasattr(fn, "_service_config"):
        return (False, None)

    from .pool import ensure_deployed, get_or_create_pool

    config = fn._service_config
    mesh = get_current_mesh()

    if mesh is None:
        return (False, None)

    pool = get_or_create_pool(fn, config, mesh)
    await ensure_deployed(pool)
    instance = pool.get_next()

    result = await _execute_with_profile(
        fn.__name__, lambda: instance.execute(*args, **kwargs)
    )

    return (True, result)


# --------------------------------------------------------------------------- #
# Actor execution hook
# --------------------------------------------------------------------------- #


async def dynamic_service_hook(
    op: Any,
    args: tuple,
    kwargs: dict,
) -> tuple[bool, Any]:
    """Hook for handling ServiceOp execution with proper access to op object.

    Args:
        op: The Op or ServiceOp being executed
        args: Resolved positional arguments
        kwargs: Resolved keyword arguments

    Returns:
        (handled, result) tuple
    """
    from .op import ServiceOp

    if not isinstance(op, ServiceOp):
        return (False, None)

    from ..resources import require_mesh
    from .pool import ensure_deployed, get_or_create_handle_pool

    service_handle = op.service_handle
    method_name = op.method_name

    mesh = require_mesh(f"Service '{service_handle.service_class.__name__}'")

    pool = get_or_create_handle_pool(service_handle, mesh)
    await ensure_deployed(pool)
    service_inst = pool.get_next()

    import asyncio

    async def call_service_method():
        instance = service_inst.get_service()
        if instance is None:
            raise RuntimeError(
                f"Cannot access service instance for {service_handle.service_class.__name__}"
            )

        method = getattr(instance, method_name)
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)

    result = await _execute_with_profile(
        f"{service_handle.service_class.__name__}.{method_name}",
        call_service_method,
    )

    return (True, result)


# --------------------------------------------------------------------------- #
# Public registration helpers
# --------------------------------------------------------------------------- #


def register_distributed_hooks() -> None:
    """Register distributed execution hooks explicitly."""
    register_hook(distributed_execution_hook)
    register_hook(dynamic_service_hook)


def unregister_distributed_hooks() -> None:
    """Unregister distributed execution hooks explicitly."""
    unregister_hook(distributed_execution_hook)
