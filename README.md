# ThinkAgain

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/BDHU/thinkagain)

A minimal, debuggable framework for async-first AI pipelines. Write small async
functions, wrap them in `Node` objects, and chain them through a lazy `Context`
that only runs when you need results. When pipelines need stateful helpers
(LLMs, retrievers, tools) you can deploy them as replica pools that run locally
or behind a gRPC server.

## Why ThinkAgain?

- **Declarative** – Build pipelines by chaining `Node` objects in plain Python
- **Lazy** – `Context` collects pending nodes and executes them on demand
- **Observable** – `ctx.metadata` records timing for each node
- **Minimal** – Small surface area, no DSLs or schedulers
- **Async-first** – Async nodes with sync and async entrypoints (`run` / `arun`)
- **Replica-aware** – Optional distributed runtime for stateful service pools

## Core Concepts

- **`Context`** – Dict-like state that records pending nodes and metadata.
- **`Node` / `@node`** – Wrap an `async def` so it can be lazily chained.
- **`run` / `arun`** – Helpers that normalize inputs and materialize pending nodes.
- **`replica`** – Decorator that turns a class into a managed pool of workers.
- **`distributed.runtime`** – Context manager that deploys replicas on local or gRPC backends.

## Installation

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

## Quick Start

```python
from thinkagain import Context, node, run

@node
async def retrieve(ctx: Context) -> Context:
    ctx.set("documents", ["doc1", "doc2"])
    return ctx

@node
async def generate(ctx: Context) -> Context:
    docs = ctx.get("documents", [])
    ctx.set("answer", f"Answer based on {docs}")
    return ctx

def pipeline(ctx: Context) -> Context:
    ctx = retrieve(ctx)
    ctx = generate(ctx)
    return ctx

result = run(pipeline, {"query": "What is ML?"})
print(result.get("answer"))
print(result.metadata.node_latencies)
```

`Context` materializes pending nodes whenever you call `run`, `arun`, `ctx.get`,
`await ctx`, or otherwise read/write actual values, so normal Python control
flow (`if`, `while`, recursion) just works.

## Distributed Replica Pools

Need a stateful helper (LLM, vector store, tool adapter)? Decorate the class
with `@replica` and let ThinkAgain manage the pool.

```python
from thinkagain import node, replica, distributed

@replica(n=2)
class FakeLLM:
    def __init__(self, prefix="Bot"):
        self.prefix = prefix

    def invoke(self, prompt: str) -> str:
        return f"{self.prefix}: {prompt}"

@node
async def call_llm(ctx):
    llm = FakeLLM.get()
    ctx.set("reply", llm.invoke(ctx.get("prompt")))
    return ctx

def pipeline(ctx):
    return call_llm(ctx)

with distributed.runtime():
    result = run(pipeline, {"prompt": "Hello"})
    print(result.get("reply"))
```

For remote deployments run the bundled gRPC server next to your replica classes
and call `distributed.runtime(backend="grpc", address="host:port")`.

## Examples

Run the declarative demo to see conditional branches and loops:

```bash
python examples/demo.py
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) – runtime, context, and replica layers
- [DESIGN.md](DESIGN.md) – execution model and control-flow patterns
- [examples/](examples/) – working demos

## License

Apache 2.0 – see [LICENSE](LICENSE)
