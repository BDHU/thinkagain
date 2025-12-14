"""Entry point for running: python -m thinkagain.serve.openai"""

import os

import uvicorn

from thinkagain import END, Context, Graph, Executable

from .serve_completion import GraphRegistry, create_app


class ProcessQuery(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.processed_query = ctx.user_query.strip()
        return ctx


class GenerateResponse(Executable):
    async def arun(self, ctx: Context) -> Context:
        query = ctx.get("processed_query", ctx.user_query)
        ctx.response = (
            f"I received your message: '{query}'\n\n"
            "This is a mock response from ThinkAgain."
        )
        return ctx


def build_demo_graph() -> Graph:
    graph = Graph(name="demo")
    graph.add("process", ProcessQuery())
    graph.add("generate", GenerateResponse())
    graph.set_entry("process")
    graph.edge("process", "generate")
    graph.edge("generate", END)
    return graph


if __name__ == "__main__":
    registry = GraphRegistry()
    registry.register("demo", build_demo_graph().compile(), set_default=True)
    app = create_app(registry)

    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
        print(f"Warning: Invalid PORT '{port_str}', using default 8000.")
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host

    print("Starting demo server...")
    print(f"Endpoint: http://{display_host}:{port}/v1/chat/completions")
    print(f"Docs: http://{display_host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)
