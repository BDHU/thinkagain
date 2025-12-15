"""Runtime utilities for node execution."""

import asyncio
from typing import Any

from .context import Context


async def _invoke(node: Any, ctx: Context) -> Context:
    """Execute a node and return the result context.

    Supports:
    - Objects with arun() method (Executable subclasses)
    - Objects with astream() method (consumes stream, returns final)
    - Async functions
    - Sync functions (run in thread)
    """
    if hasattr(node, "arun"):
        return await node.arun(ctx)

    if hasattr(node, "astream"):
        result = ctx
        async for result in node.astream(ctx):
            pass
        return result

    if asyncio.iscoroutinefunction(node):
        return await node(ctx)

    return await asyncio.to_thread(node, ctx)
