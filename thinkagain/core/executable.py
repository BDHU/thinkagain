"""
Base class for executable components.

Executables transform Context and can be wrapped in Nodes for declarative pipelines.
"""

from typing import AsyncIterator, Optional

from .context import Context


class Executable:
    """
    Base class for components that transform Context.

    Subclass and implement arun() for your logic.

    Example:
        class MyWorker(Executable):
            async def arun(self, ctx: Context) -> Context:
                ctx.result = await process(ctx.input)
                return ctx

    Usage with declarative API:
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

    async def astream(self, ctx: Context) -> AsyncIterator[Context]:
        """Stream context updates. Override for incremental output."""
        result = await self.arun(ctx)
        yield result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
