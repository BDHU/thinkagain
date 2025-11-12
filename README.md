# ThinkAgain

ThinkAgain is a minimal, debuggable agent framework for building explicit pipelines and computation graphs. It captures execution plans before they run so you can reason about complex control flow without all the hidden state most orchestration libraries introduce.

## Highlights

- **Explicit control flow** – compose workers with `>>`, Switch, Conditional, Loop, and full computation graphs (including cycles).
- **Dual sync/async APIs** – every worker can implement `__call__` and `acall`, and pipelines/graphs mirror that with `.run()` and `.arun()`.
- **Deterministic state container** – a single `Context` instance flows through the system, carrying data, metadata, and an execution history you can inspect.
- **Debuggable by default** – visualize the plan with `Pipeline.visualize()`/`Graph.to_dict()`, or replay any run by reading `ctx.history`.
- **Tiny mental model** – just a handful of primitives plus Python; no DSL, runtime, or remote control plane.

## Installation

Install the latest release from PyPI:

```bash
pip install thinkagain
```

To contribute or experiment against the local sources, use an editable install:

```bash
pip install -e .
```

## Quick Start

```python
from thinkagain import Context, Worker

class VectorDB(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.documents = self.search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

    async def acall(self, ctx: Context) -> Context:
        ctx.documents = await self.async_search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

# Compose workers with >> and run synchronously
pipeline = vector_db >> reranker >> generator
ctx = pipeline.run(Context(query="What is ML?"))

# Or run the same pipeline asynchronously
ctx = await pipeline.arun(Context(query="What is ML?"))

print(ctx.answer)
print(ctx.history)
```

## Build Workflows Your Way

### Declarative pipelines

```python
from thinkagain import Context

pipeline = retrieve >> rerank >> generate
ctx = pipeline.run(Context(query="agent evaluation"))
```

### Branching and retries

```python
from thinkagain import Switch, Loop

pipeline = (
    retrieve
    >> Switch(name="quality_check")
        .case(lambda ctx: len(ctx.documents) >= 3, rerank)
        .set_default(web_search >> rerank)
    >> Loop(
        name="refine_query",
        condition=lambda ctx: ctx.quality < 0.8,
        body=refine >> generate >> critique,
        max_iterations=3,
    )
    >> generate
)
```

### Graphs with cycles

```python
from thinkagain import Graph, END, Context

graph = Graph(name="self_correcting_rag")
graph.add_node("retrieve", RetrieveWorker())
graph.add_node("generate", GenerateWorker())
graph.add_node("critique", CritiqueWorker())
graph.add_node("refine", RefineWorker())

graph.set_entry("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edge(
    "critique",
    route=lambda ctx: "done" if ctx.quality >= 0.8 else "refine",
    paths={"done": END, "refine": "refine"},
)
graph.add_edge("refine", "retrieve")  # Cycle!

result = graph.run(Context(query="What is ML?"))
```

## Debugging & Introspection

- `Context.history` records every log message emitted by workers and control-flow nodes.
- `ctx.to_dict()` (or duck-typing with `ctx["key"]`) exposes the exact state passed between stages.
- `Pipeline.visualize()` renders an ASCII tree; `Graph.to_dict()` gives a machine-readable plan.
- `examples/graph_debugging.py` shows step-by-step execution and state inspection.

## Examples

```bash
# Pipelines: sync + async + branching + retry loop
python examples/pipeline_examples.py

# Graphs with cycles / self-correcting RAG
python examples/graph_examples.py

# Interactive graph debugging walkthrough
python examples/graph_debugging.py
```

## Documentation

See `DESIGN.md` for a deeper dive into the architecture, control-flow primitives, and future roadmap. The `thinkagain/core` package contains the minimal source code that powers everything in this repo.

## License

ThinkAgain is distributed under the Apache 2.0 License (see `LICENSE`).
