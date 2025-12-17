<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/BDHU/thinkagain/main/assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
</div>

---

A minimal, debuggable framework for building explicit, async-first AI pipelines.
ThinkAgain focuses on **declarative composition**: you write small async
functions, wrap them in `Node` objects, and chain them through a lazy `Context`
that only executes when you need results.

## Why ThinkAgain?

- **Declarative** – Build pipelines by chaining `Node` objects in plain Python
- **Lazy** – `Context` collects pending nodes and executes them on demand
- **Debuggable** – You control when materialization happens and where to inspect
- **Simple** – Just a few classes and async functions; no DSLs or heavy runtime
- **Async-first** – Async nodes with sync and async entrypoints (`run` / `arun`)

## Core Concepts

- **`@node`** – A decorator to create a reusable, chainable pipeline component from an `async` function.
- **`Node`** – The class that wraps a component to enable lazy chaining. `@node` is a shortcut for this.
- **`Context`** – A dict-like container that flows through the pipeline, holding state and enabling lazy execution.
- **`run`** – A helper to execute a pipeline synchronously.
- **`arun`** – A helper to execute a pipeline asynchronously.

## Installation

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

## Quick Start

**Define your nodes:**

```python
from thinkagain import Context, node, run

@node
async def retrieve(ctx: Context) -> Context:
    # populate documents based on query
    ctx.set("documents", ["doc1", "doc2"])
    return ctx

@node
async def generate(ctx: Context) -> Context:
    docs = ctx.get("documents", [])
    ctx.set("answer", f"Answer based on {docs}")
    return ctx
```

**Declare a pipeline with Nodes:**

```python
def pipeline(ctx: Context) -> Context:
    # Chaining is lazy: this only records pending nodes
    ctx = retrieve(ctx)
    ctx = generate(ctx)
    return ctx

result = run(pipeline, {"query": "What is ML?"})
print(result.get("answer"))
```

**Lazy materialization and control flow:**

```python
from thinkagain import arun

def conditional_pipeline(ctx: Context) -> Context:
    ctx = retrieve(ctx)
    ctx = generate(ctx)
    # Nothing has run yet – nodes are pending on the Context
    return ctx

ctx = await arun(conditional_pipeline, {"query": "What is ML?"})

if ctx.get("answer", "").startswith("Answer based on"):
    # You can now branch on computed values
    ...
else:
    ...
```

## Examples

Run the declarative demo to see conditional branches and loops:

```bash
python examples/demo.py
```

## Notes on Older Documentation

Earlier versions of ThinkAgain exposed a richer `Graph` API with visualization,
compiled graphs, and an OpenAI-compatible server. The current core library is
intentionally much smaller and focuses on the declarative `Node` + `run`
pipeline model shown above. The architectural docs in `ARCHITECTURE.md` and
`DESIGN.md` describe that earlier design and are kept as historical notes.

- [ARCHITECTURE.md](ARCHITECTURE.md) – design rationale
- [DESIGN.md](DESIGN.md) – control-flow primitives and roadmap
- [examples/](examples/) – working demos

## License

Apache 2.0 – see [LICENSE](LICENSE)
