"""Runtime utilities for node execution."""

import asyncio
from typing import Any

from .context import Context


async def _invoke(node: Any, ctx: Context) -> Context:
    """Execute a node and return the result context."""
    if hasattr(node, "arun"):
        return await node.arun(ctx)

    if asyncio.iscoroutinefunction(node):
        return await node(ctx)

    return await asyncio.to_thread(node, ctx)
