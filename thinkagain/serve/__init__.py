"""
ThinkAgain OpenAI-Compatible Server
====================================

Reusable components for creating OpenAI-compatible API servers
that wrap ThinkAgain graph execution.

Usage:
    from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry
    from thinkagain import Graph, Worker, Context, END

    # Define your worker
    class MyWorker(Worker):
        def __call__(self, ctx: Context) -> Context:
            ctx.response = "Your response here"
            return ctx

    # Build graph
    graph = Graph(name="my_graph")
    graph.add_node("worker", MyWorker())
    graph.add_edge("worker", END)

    # Create app
    registry = GraphRegistry()
    registry.register("my-model", graph, set_default=True)
    app = create_app(registry)

    # Run with uvicorn
    # uvicorn my_module:app

See thinkagain/serve/README.md for full documentation.
"""

from .openai.serve_completion import create_app, GraphRegistry

__all__ = ["create_app", "GraphRegistry"]
