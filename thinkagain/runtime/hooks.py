"""Execution hooks for pluggable extensions.

Provides a lightweight hook system for extending executor behavior without
tight coupling. Extensions (like distributed execution and actor execution)
can register hooks to intercept and modify execution flow.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Protocol

__all__ = [
    "ExecutionHook",
    "register_hook",
    "unregister_hook",
    "get_hooks",
    "clear_hooks",
    "register_distributed_hooks",
    "unregister_distributed_hooks",
    "dynamic_actor_hook",
]


class ExecutionHook(Protocol):
    """Protocol for execution hooks.

    Hooks can intercept function execution and either:
    1. Return (True, result) to short-circuit execution with a custom result
    2. Return (False, None) to continue with normal execution
    """

    async def __call__(
        self,
        task: Any | Callable,
        args: tuple,
        kwargs: dict,
        node_id: int | None = None,
    ) -> tuple[bool, Any]:
        """Intercept function execution.

        Args:
            task: Task object or function being executed
            args: Positional arguments
            kwargs: Keyword arguments
            node_id: Optional node ID for context

        Returns:
            (handled, result) tuple:
            - If handled=True, result is used instead of calling fn
            - If handled=False, normal execution continues
        """
        ...


# Global hook registry (order matters - first match wins)
_hooks: list[ExecutionHook] = []


def register_hook(hook: ExecutionHook) -> None:
    """Register an execution hook.

    Hooks are checked in registration order. The first hook that returns
    (True, result) will short-circuit execution.

    Args:
        hook: Hook to register
    """
    if hook not in _hooks:
        _hooks.append(hook)


def unregister_hook(hook: ExecutionHook) -> None:
    """Unregister an execution hook.

    Args:
        hook: Hook to remove
    """
    if hook in _hooks:
        _hooks.remove(hook)


def get_hooks() -> list[ExecutionHook]:
    """Get all registered hooks (for testing/debugging)."""
    return _hooks.copy()


def clear_hooks() -> None:
    """Clear all hooks (for testing)."""
    _hooks.clear()


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
        profiling.record_replicate_call(name, duration=duration)


async def distributed_execution_hook(
    fn: Any,
    args: tuple,
    kwargs: dict,
    node_id: int | None = None,
) -> tuple[bool, Any]:
    """Hook to handle distributed execution of replicated classes.

    Args:
        fn: Class or callable being executed
        args: Positional arguments
        kwargs: Keyword arguments
        node_id: Optional node ID

    Returns:
        (handled, result) tuple:
        - If fn has _replica_config and mesh is active: (True, result)
        - Otherwise: (False, None) to continue with normal execution
    """
    # Check if class is a replica
    if not hasattr(fn, "_replica_config"):
        return (False, None)

    from ..resources import get_current_mesh
    from .pool import ensure_deployed, get_or_create_pool

    config = fn._replica_config
    mesh = get_current_mesh()

    if mesh is None:
        return (False, None)

    pool = get_or_create_pool(fn, config, mesh)
    await ensure_deployed(pool)
    replica = pool.get_next()

    result = await _execute_with_profile(
        fn.__name__, lambda: replica.execute(*args, **kwargs)
    )

    return (True, result)


# --------------------------------------------------------------------------- #
# Actor execution hook
# --------------------------------------------------------------------------- #


async def dynamic_actor_hook(
    task: Any,
    args: tuple,
    kwargs: dict,
) -> tuple[bool, Any]:
    """Hook for handling ActorTask execution with proper access to task object.

    Args:
        task: The Task or ActorTask being executed
        args: Resolved positional arguments
        kwargs: Resolved keyword arguments

    Returns:
        (handled, result) tuple
    """
    from .task import ActorTask

    if not isinstance(task, ActorTask):
        return (False, None)

    from ..resources import get_current_mesh
    from .pool import ensure_deployed, get_or_create_handle_pool

    actor_handle = task.actor_handle
    method_name = task.method_name

    mesh = get_current_mesh()
    if mesh is None:
        raise RuntimeError(
            f"Actor {actor_handle._replica_class.__name__} requires a mesh context. "
            "Use 'with Mesh([...]):' to create a mesh before calling actor methods."
        )

    pool = get_or_create_handle_pool(actor_handle.replica_handle, mesh)
    await ensure_deployed(pool)
    replica = pool.get_next()

    import asyncio

    async def call_actor_method():
        if hasattr(replica, "state") and replica.state is not None:
            instance = replica.state
        elif hasattr(replica, "fn") and replica.fn is not None:
            instance = replica.fn
            if not hasattr(replica, "state") or replica.state is None:
                replica.state = instance
        else:
            raise RuntimeError(
                f"Cannot access actor instance for {actor_handle._replica_class.__name__}"
            )

        method = getattr(instance, method_name)
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)

    result = await _execute_with_profile(
        f"{actor_handle._replica_class.__name__}.{method_name}",
        call_actor_method,
    )

    return (True, result)


# --------------------------------------------------------------------------- #
# Public registration helpers
# --------------------------------------------------------------------------- #


def register_distributed_hooks() -> None:
    """Register distributed execution hooks explicitly."""
    register_hook(distributed_execution_hook)


def unregister_distributed_hooks() -> None:
    """Unregister distributed execution hooks explicitly."""
    unregister_hook(distributed_execution_hook)
