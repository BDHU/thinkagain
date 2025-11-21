<div align="center" id="sglangtop">
<img src="assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)

</div>

---

ThinkAgain is a minimal, debuggable agent framework for building explicit pipelines and computation graphs. It captures execution plans before they run so you can reason about complex control flow without all the hidden state most orchestration libraries introduce.

## Highlights

- **Graph-first architecture** – everything inherits from `Executable`, so workers, graphs, and pipelines compose naturally with `>>`.
- **Async core, sync friendly** – all executables expose `arun(ctx)`; synchronous calls simply wrap that single code path.
- **Deterministic Context** – one `Context` object carries state, metadata, and execution history through the system.
- **First-class introspection** – `Graph.visualize()`, `Graph.to_dict()`, `graph.compile()`, and `ctx.history` reveal plans before and after they run.
- **Minimal surface area** – just Python classes; no DSLs, no sidecar runtime, and no hidden orchestration layers.

## Architecture at a Glance

ThinkAgain reduces the mental model to a handful of building blocks:

- **Executable** – base interface defining `__call__`, `arun`, and composition via `__rshift__`.
- **Worker** – your business logic; implement either sync or async and the framework handles the rest.
- **Graph** – the scheduler that stores nodes, direct edges, and conditional edges (cycles are just edges that point backwards). The `>>` operator creates sequential graphs automatically.
- **Context** – a deterministic, dict-like container with a chronological `history` of every log emission.

All nodes are Executables, so subgraphs plug directly into larger graphs, and sequential flows stay ergonomic with the `>>` operator.

## Installation

Install the latest release from PyPI:

```bash
pip install thinkagain
```

To contribute or experiment against the local sources, use an editable install:

```bash
pip install -e .
```

If you are using [uv](https://github.com/astral-sh/uv), simply do:

```bash
uv add thinkagain
```

## Quick Start

```python
from thinkagain import Context, Worker, Graph

class VectorDB(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.documents = self.search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

    async def arun(self, ctx: Context) -> Context:
        ctx.documents = await self.async_search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

vector_db = VectorDB()

# Compose workers with >> to form a sequential pipeline
pipeline = vector_db >> Reranker() >> Generator()  # assume these are Worker subclasses
ctx = pipeline(Context(query="What is ML?"))

# Graphs make routing explicit
graph = Graph(name="rag")
graph.add_node("retrieve", vector_db)
graph.add_node("rerank", Reranker())
graph.add_node("generate", Generator())
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "generate")

# Async execution is the canonical path
ctx = await graph.arun(Context(query="What is ML?"))
print(ctx.answer)
print(ctx.history)  # chronological log of every node
```

### Wrapping plain async functions

Use `@async_worker` to turn a simple coroutine into a first-class worker without creating a subclass:

```python
from thinkagain import async_worker, Context

@async_worker
async def fetch(ctx: Context) -> Context:
    ctx.result = await ctx.client.get(ctx.query)
    return ctx

pipeline = fetch >> postprocess  # postprocess can be another Worker
ctx = await pipeline.arun(Context(query="hello"))
```

## Build Workflows Your Way

### Sequential pipelines

```python
from thinkagain import Context

pipeline = retrieve >> rerank >> generate
ctx = pipeline(Context(query="agent evaluation"))

# async execution
ctx = await pipeline.arun(Context(query="agent evaluation"))
```

### Graphs with routing and cycles

```python
from thinkagain import Graph, END, Context

graph = Graph(name="self_correcting_rag")
graph.add_node("retrieve", RetrieveWorker())
graph.add_node("generate", GenerateWorker())
graph.add_node("critique", CritiqueWorker())
graph.add_node("refine", RefineWorker())

graph.set_entry("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_conditional_edge(
    "generate",
    route=lambda ctx: "done" if ctx.quality >= 0.8 else "critique",
    paths={"done": END, "critique": "critique"},
)
graph.add_edge("critique", "refine")
graph.add_edge("refine", "retrieve")  # Cycle back for another pass

result = await graph.arun(Context(query="What is ML?"))
```

### Subgraphs compose naturally

```python
research = build_research_agent()  # returns Graph
writer = build_writing_agent()     # returns Graph

coordinator = Graph(name="coordinator")
coordinator.add_node("research", research)
coordinator.add_node("write", writer)
coordinator.add_edge("research", "write")
```

## Debugging & Introspection

- `Context.history` records every log message emitted by workers and graph nodes.
- `ctx.to_dict()` (or duck-typing with `ctx["key"]`) shows the exact state shuttled between stages.
- `graph.stream(ctx)` (or `compiled.stream(ctx)`) yields events as each node completes so you can surface partial results.
- `Graph.visualize()` renders a Mermaid diagram; `Graph.to_dict()` and `graph.compile()` produce machine-readable plans.
- `examples/minimal_demo.py` prints both the execution logs and a Mermaid graph so you can watch the state evolve.

## Examples

```bash
# One-file tour of pipelines, graphs, and compile()
python examples/minimal_demo.py
```

## Documentation

See `ARCHITECTURE.md` for the graph-first rationale and `DESIGN.md` for the control-flow primitives plus roadmap. The `thinkagain/core` package contains the minimal source that powers everything in this repo.

## License

ThinkAgain is distributed under the Apache 2.0 License (see `LICENSE`).
