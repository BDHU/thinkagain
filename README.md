<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/BDHU/thinkagain/main/assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
</div>

---

A minimal, debuggable agent framework for building explicit pipelines and computation graphs. ThinkAgain captures execution plans before they run so you can reason about complex control flow without hidden state.

## Why ThinkAgain?

- **Explicit** – Build graphs with `add()` and `edge()`, then `compile()` to execute
- **Transparent** – `Context` carries state and execution history; `visualize()` shows your pipeline
- **Debuggable** – Stream execution events, inspect history, and export plans as data
- **Simple** – Just Python classes; no DSLs or hidden orchestration layers
- **Async-first** – Native async support with sync wrappers when needed

## Core Concepts

- **Executable** – Base class for components; implement `arun()` (and optionally `astream()` for streaming)
- **Graph** – Builder for connecting executables with edges; supports routing and cycles
- **CompiledGraph** – Executable created by `graph.compile()`; run with `arun()` or `stream()`
- **Context** – Dict-like container that flows through your pipeline, tracking history

## Installation

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

## Quick Start

**Define your executables:**

```python
from thinkagain import Context, Executable, Graph, END

class Retriever(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.documents = await self.search(ctx.query)
        return ctx

class Generator(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.answer = await self.generate(ctx.documents)
        return ctx
```

**Build and run a graph:**

```python
graph = Graph(name="rag_pipeline")
graph.add("retrieve", Retriever())
graph.add("generate", Generator())
graph.edge("retrieve", "generate")
graph.edge("generate", END)

result = await graph.compile().arun(Context(query="What is ML?"))
print(result.answer)
```

**Graphs with routing and cycles:**

```python
graph = Graph(name="self_correcting_rag")
graph.add("retrieve", Retriever())
graph.add("generate", Generator())
graph.add("critique", Critic())

graph.set_entry("retrieve")
graph.edge("retrieve", "generate")
graph.edge("generate", "critique")
graph.edge("critique", lambda ctx: END if ctx.quality >= 0.8 else "retrieve")

result = await graph.compile().arun(Context(query="What is ML?"))
```

**Decorator syntax** for simple executables:

```python
from thinkagain import async_executable

@async_executable
async def fetch(ctx: Context) -> Context:
    ctx.data = await ctx.client.get(ctx.url)
    return ctx

graph = Graph()
graph.add("fetch", fetch)
graph.edge("fetch", END)
```

## Examples

Run the demo to see graphs and visualization:

```bash
python examples/minimal_demo.py
```

## OpenAI-Compatible Server

Optional server with OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
pip install "thinkagain[serve]"

# Start server
python -m thinkagain.serve.openai
```

See [thinkagain/serve/README.md](thinkagain/serve/README.md) for details.

## Learn More

- [ARCHITECTURE.md](ARCHITECTURE.md) – design rationale
- [DESIGN.md](DESIGN.md) – control-flow primitives and roadmap
- [examples/](examples/) – working demos

## License

Apache 2.0 – see [LICENSE](LICENSE)
