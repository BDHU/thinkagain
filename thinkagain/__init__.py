"""
thinkagain - A minimal framework for declarative AI pipelines.

Example:
    from thinkagain import Node, run, Executable, Context

    class Retriever(Executable):
        async def arun(self, ctx: Context) -> Context:
            ctx.docs = ["doc1", "doc2"]
            return ctx

    class Generator(Executable):
        async def arun(self, ctx: Context) -> Context:
            ctx.answer = f"Answer based on {ctx.docs}"
            return ctx

    retrieve = Node(Retriever())
    generate = Node(Generator())

    async def pipeline(ctx):
        ctx = await retrieve(ctx)
        ctx = await generate(ctx)
        return ctx

    result = await run(pipeline, {"query": "test"})

Nodes chain lazily - use `await ctx` to materialize before accessing values:

    async def pipeline(ctx):
        ctx = await node_a(ctx)
        ctx = await node_b(ctx)
        ctx = await ctx          # materialize here
        if ctx.score > 0.8:      # now safe to access
            ctx = await node_c(ctx)
        return ctx
"""

from .core import (
    Node,
    run,
    Context,
    Executable,
    LazyContext,
    NeedsMaterializationError,
)

__version__ = "0.3.0"

__all__ = [
    "Node",
    "run",
    "Context",
    "Executable",
    "LazyContext",
    "NeedsMaterializationError",
]
