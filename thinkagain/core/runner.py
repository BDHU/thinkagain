"""Run declarative pipelines."""

from typing import Callable

from .context import Context


def run(fn: Callable[[Context], Context], ctx: dict | None = None) -> Context:
    """Run pipeline synchronously and return final context."""
    result = fn(Context(ctx))
    if result.is_pending:
        result._sync()
    return result


async def arun(fn: Callable[[Context], Context], ctx: dict | None = None) -> Context:
    """Run pipeline asynchronously - use this for concurrent request handling."""
    result = fn(Context(ctx))
    if result.is_pending:
        await result._async_sync()
    return result
