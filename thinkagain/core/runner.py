"""Run declarative pipelines."""

from typing import Awaitable, Callable

from .context import Context
from .lazy import LazyContext


async def run(fn: Callable[[LazyContext], Awaitable[LazyContext]], ctx: dict | None = None) -> Context:
    """Run pipeline and return final context."""
    result = await fn(LazyContext(ctx or {}))
    if result.is_pending:
        result = await result
    return Context(result.data)
