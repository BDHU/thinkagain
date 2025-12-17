"""Run declarative pipelines."""

from typing import Callable

from .context import Context


def run(fn: Callable[[Context], Context], ctx: dict | None = None) -> Context:
    """Run pipeline synchronously and return final context."""
    return fn(Context(ctx)).materialize()


async def arun(fn: Callable[[Context], Context], ctx: dict | None = None) -> Context:
    """Run pipeline asynchronously."""
    return await fn(Context(ctx)).amaterialize()
