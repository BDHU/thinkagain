# ThinkAgain

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/BDHU/thinkagain?utm_source=badge)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A minimal, debuggable framework for async-first AI pipelines. Write small async
functions, wrap them in `Node` objects, and chain them through a lazy `Context`
that only runs when you need results. Contexts track dependencies via
back-pointers, enabling DAG execution with automatic deduplication for fanout
patterns. When pipelines need stateful helpers (LLMs, retrievers, tools) you
can deploy them as replica pools that run locally or behind a gRPC server.

## Why ThinkAgain?

- **Declarative** – Build pipelines by chaining `Node` objects in plain Python
- **Lazy** – `Context` collects pending nodes and executes them on demand
- **DAG Execution** – Back-pointer tracking enables fanout with automatic deduplication
- **Multi-Input Nodes** – Nodes can accept multiple context arguments
- **Minimal** – Small surface area, no DSLs or schedulers
- **Async-first** – Async nodes with sync and async entrypoints (`run` / `arun`)
- **Replica-aware** – Optional distributed runtime for stateful service pools

## Core Concepts

- **`Context`** – Wrapper around user data that tracks parent contexts via back-pointers.
- **`Node` / `@node`** – Wrap an `async def` so it can be lazily chained. Nodes can accept multiple inputs.
- **`run` / `arun`** – Helpers that normalize inputs and materialize pending nodes via DAG traversal.
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
from dataclasses import dataclass
from thinkagain import node, run

@dataclass
class State:
    query: str
    documents: list[str] | None = None
    answer: str = ""

@node
async def retrieve(s: State) -> State:
    return State(query=s.query, documents=["doc1", "doc2"])

@node
async def generate(s: State) -> State:
    docs = s.documents or []
    return State(query=s.query, documents=docs, answer=f"Answer based on {docs}")

def pipeline(ctx):
    ctx = retrieve(ctx)
    ctx = generate(ctx)
    return ctx

result = run(pipeline, State(query="What is ML?"))
print(result.data.answer)
```

Nodes receive and return plain Python values (dataclasses, dicts, etc.) which
are automatically wrapped in `Context`. The context materializes pending nodes
whenever you access `ctx.data`, call `run`, `arun`, or `await ctx`, so normal
Python control flow (`if`, `while`, recursion) just works.

## Distributed Replica Pools

Need a stateful helper (LLM, vector store, tool adapter)? Decorate the class
with `@replica` and let ThinkAgain manage the pool.

```python
from dataclasses import dataclass
from thinkagain import node, replica, distributed, run

@dataclass
class ChatState:
    prompt: str
    reply: str = ""

@replica(n=2)
class FakeLLM:
    def __init__(self, prefix="Bot"):
        self.prefix = prefix

    def invoke(self, prompt: str) -> str:
        return f"{self.prefix}: {prompt}"

@node
async def call_llm(s: ChatState) -> ChatState:
    llm = FakeLLM.get()
    return ChatState(prompt=s.prompt, reply=llm.invoke(s.prompt))

def pipeline(ctx):
    return call_llm(ctx)

with distributed.runtime():
    result = run(pipeline, ChatState(prompt="Hello"))
    print(result.data.reply)
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
