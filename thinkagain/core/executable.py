"""Base class for executable components."""

from typing import Optional

from .context import Context


class Executable:
    """
    Base class for components that transform Context.

    Example:
        class MyWorker(Executable):
            async def arun(self, ctx: Context) -> Context:
                ctx.result = await process(ctx.input)
                return ctx

    Usage:
        worker = Node(MyWorker())

        async def pipeline(ctx):
            ctx = await worker(ctx)
            return ctx
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name if name is not None else self.__class__.__name__.lower()

    async def arun(self, ctx: Context) -> Context:
        """Execute and return modified context. Subclasses must implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement arun()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
