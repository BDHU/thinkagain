"""
Example: Basic OpenAI Server Setup
==================================

Shows how to create a simple OpenAI-compatible server with a custom executable.

Run with:
    python examples/serve_openai_demo.py
    # or
    uvicorn examples.serve_openai_demo:app --reload
"""

from thinkagain import Context, Executable, Graph, END
from thinkagain.serve.openai.serve_completion import create_app, GraphRegistry


class MyExecutable(Executable):
    """Your custom executable - replace with your LLM integration."""

    async def arun(self, ctx: Context) -> Context:
        ctx.response = f"You asked: '{ctx.user_query}'\n\nThis is a custom response!"
        return ctx


# Build graph
graph = Graph(name="simple")
graph.add("worker", MyExecutable())
graph.edge("worker", END)

# Create app
registry = GraphRegistry()
registry.register("simple", graph.compile(), set_default=True)
app = create_app(registry)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
