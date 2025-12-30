"""Run declarative pipelines."""

from typing import Any, Callable, ContextManager

from .context import Context


def chain(*steps: Callable[[Context], Context]) -> Callable[[Context], Context]:
    """Build a linear pipeline from a sequence of steps."""

    def pipeline(ctx: Context) -> Context:
        for step in steps:
            ctx = step(ctx)
        return ctx

    return pipeline


def run(
    fn: Callable[[Context], Context],
    data: Any = None,
    *,
    context_factory: Callable[[str], ContextManager[None]] | None = None,
) -> Context:
    """Run pipeline synchronously and return final context.

    Example:
        @node
        async def add_one(x: int) -> int:
            return x + 1

        def pipeline(ctx):
            return add_one(ctx)

        result = run(pipeline, 5)
        print(result.data)  # 6

    Use context_factory for cross-cutting concerns like profiling.
    """
    ctx = fn(Context(data, context_factory=context_factory))
    _ = ctx.data
    return ctx


async def arun(
    fn: Callable[[Context], Context],
    data: Any = None,
    *,
    context_factory: Callable[[str], ContextManager[None]] | None = None,
) -> Context:
    """Run pipeline asynchronously.

    Example:
        result = await arun(pipeline, 5)
        print(result.data)  # 6

    Use context_factory for cross-cutting concerns like profiling.
    """
    ctx = fn(Context(data, context_factory=context_factory))
    await ctx
    return ctx
