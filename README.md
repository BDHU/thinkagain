<div align="center" id="sglangtop">
<img src="assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
</div>

---

A minimal, debuggable agent framework for building explicit pipelines and computation graphs. ThinkAgain captures execution plans before they run so you can reason about complex control flow without hidden state.

## Why ThinkAgain?

- **Composable** – Workers and graphs compose naturally with `>>` operator
- **Transparent** – `Context` carries state and execution history; `graph.visualize()` shows your pipeline
- **Debuggable** – Stream execution events, inspect history, and export plans as data
- **Simple** – Just Python classes; no DSLs or hidden orchestration layers
- **Async-first** – Native async support with sync wrappers when needed

## Core Concepts

- **Worker** – Implement your logic; subclass and define `__call__` or `arun`
- **Graph** – Connect workers with edges; supports routing and cycles
- **Context** – Dict-like container that flows through your pipeline, tracking history
- **Executable** – Base interface that enables `>>` composition

## Installation

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

## Quick Start

**Sequential pipelines** with the `>>` operator:

```python
from thinkagain import Context, Worker

class Retriever(Worker):
    async def arun(self, ctx: Context) -> Context:
        ctx.documents = await self.search(ctx.query)
        return ctx

pipeline = Retriever() >> Reranker() >> Generator()
ctx = await pipeline.arun(Context(query="What is ML?"))
print(ctx.answer)
```

**Graphs** with routing and cycles:

```python
from thinkagain import Graph, END

graph = Graph(name="self_correcting_rag")
graph.add_node("retrieve", Retriever())
graph.add_node("generate", Generator())
graph.add_node("critique", Critic())

graph.set_entry("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_conditional_edge(
    "generate",
    route=lambda ctx: "done" if ctx.quality >= 0.8 else "critique",
    paths={"done": END, "critique": "critique"},
)
graph.add_edge("critique", "retrieve")  # retry loop

ctx = await graph.arun(Context(query="What is ML?"))
```

**Decorator syntax** for simple workers:

```python
from thinkagain import async_worker

@async_worker
async def fetch(ctx: Context) -> Context:
    ctx.data = await ctx.client.get(ctx.url)
    return ctx

pipeline = fetch >> process
```

## Examples

Run the demo to see pipelines, graphs, and visualization:

```bash
python examples/minimal_demo.py
```

## OpenAI-Compatible Server

Optional server with OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
pip install "thinkagain[serve]"
# or with uv
uv pip install -e ".[serve]"

# Start server
python -m thinkagain.serve.openai.serve_completion
```

See [thinkagain/serve/README.md](thinkagain/serve/README.md) for details.

## Learn More

- [ARCHITECTURE.md](ARCHITECTURE.md) – design rationale
- [DESIGN.md](DESIGN.md) – control-flow primitives and roadmap
- [examples/](examples/) – working demos

## License

Apache 2.0 – see [LICENSE](LICENSE)
