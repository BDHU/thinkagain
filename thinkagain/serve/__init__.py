"""
ThinkAgain OpenAI-Compatible Server
====================================

Reusable components for creating OpenAI-compatible API servers
that wrap ThinkAgain graph execution.

Usage:
    from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry
    from thinkagain import Graph, Executable, Context, END

    class MyExecutable(Executable):
        async def arun(self, ctx: Context) -> Context:
            ctx.response = "Your response here"
            return ctx

    # Build and compile graph
    graph = Graph(name="my_graph")
    graph.add("worker", MyExecutable())
    graph.edge("worker", END)
    compiled = graph.compile()

    # Register and create app
    registry = GraphRegistry()
    registry.register("my-model", compiled, set_default=True)
    app = create_app(registry)

See thinkagain/serve/README.md for full documentation.
"""

from .openai.serve_completion import create_app, GraphRegistry

__all__ = ["create_app", "GraphRegistry"]
