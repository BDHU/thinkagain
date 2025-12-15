"""
Core components for declarative graph construction.

Example:
    from thinkagain import Node, run, Executable, Context

    class MyWorker(Executable):
        async def arun(self, ctx: Context) -> Context:
            ctx.result = "done"
            return ctx

    worker = Node(MyWorker())

    async def pipeline(ctx):
        ctx = await worker(ctx)
        return ctx

    result = await run(pipeline, {"input": "test"})
"""

from .context import Context
from .executable import Executable
from .node import Node
from .lazy import LazyContext, NeedsMaterializationError
from .runner import run

__all__ = [
    "Node",
    "run",
    "Context",
    "Executable",
    "LazyContext",
    "NeedsMaterializationError",
]
