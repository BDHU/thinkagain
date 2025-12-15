"""
Runner for declarative pipelines.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Union

from .context import Context
from .lazy import LazyContext


async def run(
    fn: Callable[[LazyContext], Awaitable[LazyContext]],
    ctx: Union[Context, dict, None] = None,
) -> Context:
    """
    Run a declarative pipeline and return the final Context.

    Args:
        fn: Async function that takes LazyContext and returns LazyContext
        ctx: Initial context (Context, dict, or None for empty)

    Returns:
        Final materialized Context

    Example:
        retrieve = Node(RetrieveDocs())
        generate = Node(GenerateAnswer())

        async def pipeline(ctx):
            ctx = await retrieve(ctx)
            ctx = await generate(ctx)
            return ctx

        result = await run(pipeline, {"query": "test"})
        print(result.answer)
    """
    # Normalize initial context
    if ctx is None:
        initial_data = {}
    elif isinstance(ctx, dict):
        initial_data = ctx
    else:
        initial_data = ctx.data

    # Create lazy context and run pipeline
    lazy_ctx = LazyContext(initial_data, [])
    result = await fn(lazy_ctx)

    # Final materialization if needed
    if result.is_pending:
        result = await result

    # Return as regular Context
    return Context(**result.data)
