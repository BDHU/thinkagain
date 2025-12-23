"""Run declarative pipelines."""

from typing import Any, Callable

from .context import Context


def run(fn: Callable[[Context], Context], data: Any = None) -> Context:
    """Run pipeline synchronously and return final context.

    Example:
        @node
        async def add_one(x: int) -> int:
            return x + 1

        def pipeline(ctx):
            return add_one(ctx)

        result = run(pipeline, 5)
        print(result.data)  # 6
    """
    return fn(Context(data)).materialize()


async def arun(fn: Callable[[Context], Context], data: Any = None) -> Context:
    """Run pipeline asynchronously.

    Example:
        result = await arun(pipeline, 5)
        print(result.data)  # 6
    """
    return await fn(Context(data)).amaterialize()
