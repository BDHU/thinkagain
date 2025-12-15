"""Runtime utilities for node execution."""

import asyncio
from typing import Any

from .context import Context


async def _invoke(node: Any, ctx: Context) -> Context:
    """Execute a node and return the result context.

    Supports:
    - Objects with astream() method (consumes stream, returns final)
    - Objects with arun() method (Executable subclasses)
    - Async functions
    - Sync functions (run in thread)
    """
    # Streaming executables: consume the stream and return the final Context
    if hasattr(node, "astream"):
        result = ctx
        async for result in node.astream(ctx):
            pass
        return result

    # Simple async executables: arun(ctx) -> Context
    if hasattr(node, "arun"):
        return await node.arun(ctx)

    # Plain async callables: node(ctx) -> awaitable[Context]
    if asyncio.iscoroutinefunction(node):
        return await node(ctx)

    # Fallback: run sync callables in a thread
    return await asyncio.to_thread(node, ctx)
