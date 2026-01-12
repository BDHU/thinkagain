"""Distributed execution hook for replicated classes.

This module provides the hook that intercepts calls to @replica classes
and routes them through the replica pool system when a mesh context is active.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable


async def _execute_with_profile(name: str, call: Callable[[], Awaitable[Any]]) -> Any:
    from ..core import profiling

    if profiling.get_profiler() is None:
        return await call()

    start_time = time.perf_counter()
    try:
        return await call()
    finally:
        duration = time.perf_counter() - start_time
        profiling.record_replicate_call(name, duration=duration)


async def distributed_execution_hook(
    fn: Any,  # Can be a replica class or regular callable
    args: tuple,
    kwargs: dict,
    node_id: int | None = None,
) -> tuple[bool, Any]:
    """Hook to handle distributed execution of replicated classes.

    Args:
        fn: Class being executed
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

    from .mesh import get_current_mesh
    from .replication.pool import ensure_deployed, get_or_create_pool

    config = fn._replica_config
    mesh = get_current_mesh()

    if mesh is None:
        # No mesh context - execute locally
        return (False, None)

    # Get or create pool for this replica class
    pool = get_or_create_pool(fn, config, mesh)

    # Ensure deployed (auto-deploy with n=1 if needed)
    await ensure_deployed(pool)

    # Get next replica (round-robin)
    replica = pool.get_next()

    # Execute on replica with optional profiling
    result = await _execute_with_profile(
        fn.__name__, lambda: replica.execute(*args, **kwargs)
    )

    return (True, result)
