"""
Simple Example: Basic OpenAI Server Setup
==========================================

This example shows how to create a simple OpenAI-compatible server
with your own custom worker.

Run with:
    python examples/serve_openai_demo.py
    # or (when running via module path)
    uvicorn examples.serve_openai_demo:app --reload
"""

from thinkagain import Context, Worker, Graph, END
from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry


class MyWorker(Worker):
    """Your custom worker - replace with your LLM integration."""

    async def arun(self, ctx: Context) -> Context:
        ctx.response = f"You asked: '{ctx.user_query}'\n\nThis is a custom response!"
        return ctx


# Build graph
graph = Graph(name="simple")
graph.add_node("worker", MyWorker())
graph.add_edge("worker", END)

# Create app
registry = GraphRegistry()
registry.register("simple", graph, set_default=True)
app = create_app(registry)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
