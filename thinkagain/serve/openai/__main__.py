"""Entry point for running: python -m thinkagain.serve.openai"""

import os

import uvicorn

from thinkagain import END, Context, Graph, Worker

from .serve_completion import GraphRegistry, create_app


class ProcessQuery(Worker):
    async def arun(self, ctx: Context) -> Context:
        ctx.processed_query = ctx.user_query.strip()
        return ctx


class GenerateResponse(Worker):
    async def arun(self, ctx: Context) -> Context:
        query = ctx.get("processed_query", ctx.user_query)
        ctx.response = (
            f"I received your message: '{query}'\n\n"
            "This is a mock response from ThinkAgain."
        )
        return ctx


def build_demo_graph() -> Graph:
    graph = Graph(name="demo")
    graph.add_node("process", ProcessQuery())
    graph.add_node("generate", GenerateResponse())
    graph.set_entry("process")
    graph.add_edge("process", "generate")
    graph.add_edge("generate", END)
    return graph


if __name__ == "__main__":
    registry = GraphRegistry()
    registry.register("demo", build_demo_graph(), set_default=True)
    app = create_app(registry)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host

    print("Starting demo server...")
    print(f"Endpoint: http://{display_host}:{port}/v1/chat/completions")
    print(f"Docs: http://{display_host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)
