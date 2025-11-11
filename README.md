# ThinkAgain

A simple, minimal framework for building agent pipelines with explicit control.

## Features

- **Dual API**: Run pipelines synchronously with `.run()` or asynchronously with `.arun()`
- **Explicit Control Flow**: Pipeline with Switch, Loop, and Conditional
- **Complex Workflows**: Graph with cycles for self-correcting agents
- **Full Debuggability**: Step-by-step execution with state inspection
- **Clean Composition**: Use `>>` operator to chain workers

## Quick Start

```python
from thinkagain import Context, Worker

# Create a worker with both sync and async support
class VectorDB(Worker):
    def __call__(self, ctx: Context) -> Context:
        ctx.documents = self.search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

    async def acall(self, ctx: Context) -> Context:
        ctx.documents = await self.async_search(ctx.query)
        ctx.log(f"Retrieved {len(ctx.documents)} docs")
        return ctx

# Build and run pipeline synchronously
pipeline = vector_db >> reranker >> generator
ctx = pipeline.run(Context(query="What is ML?"))

# Or run asynchronously
ctx = await pipeline.arun(Context(query="What is ML?"))

# Inspect results at any point
print(ctx.answer)
print(ctx.history)
```

## Core Components

**Context** - Container that holds state and tracks history as it passes through workers

**Worker** - Base class for processing units with dual API: `__call__()` for sync, `acall()` for async

**Pipeline** - Linear workflows with conditionals and loops

**Graph** - Complex workflows with cycles (e.g., self-correcting agents)

## Pipeline Examples

```python
from thinkagain import Pipeline, Switch, Loop

# Simple composition
pipeline = vector_db >> reranker >> generator
result = pipeline.run(Context(query="What is ML?"))

# Conditional branching
pipeline = (
    retrieve
    >> Switch(name="quality_check")
        .case(lambda ctx: len(ctx.documents) >= 3, rerank)
        .set_default(web_search >> rerank)
    >> generate
)

# Loops for iterative refinement
pipeline = (
    retrieve
    >> Loop(
        condition=lambda ctx: len(ctx.documents) < 2,
        body=refine_query >> retrieve,
        max_iterations=3
    )
    >> generate
)

# Concurrent execution (async benefit!)
tasks = [pipeline.arun(Context(query=q)) for q in queries]
results = await asyncio.gather(*tasks)
```

## Graph with Cycles

For workflows that need to loop back (e.g., self-correcting agents):

```python
from thinkagain import Graph, END

graph = Graph(name="self_correcting_rag")

# Add nodes
graph.add_node("retrieve", RetrieveWorker())
graph.add_node("generate", GenerateWorker())
graph.add_node("critique", CritiqueWorker())
graph.add_node("refine", RefineWorker())

# Define flow with cycle
graph.set_entry("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edge(
    "critique",
    route=lambda ctx: "done" if ctx.quality >= 0.8 else "refine",
    paths={"done": END, "refine": "refine"}
)
graph.add_edge("refine", "retrieve")  # Cycle!

# Execute
result = graph.run(Context(query="What is ML?"))
```

## Examples

```bash
# Pipeline examples (sync, async, Switch, Loop, concurrent)
python examples/pipeline_examples.py

# Graph with cycles (self-correcting RAG)
python examples/graph_examples.py

# Graph debugging (step-by-step execution)
python examples/graph_debugging.py
```
