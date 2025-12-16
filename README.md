<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/BDHU/thinkagain/main/assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
</div>

---

A minimal, debuggable framework for building explicit, async-first AI pipelines.
ThinkAgain focuses on **declarative composition**: you define reusable components
(`Executable`), wrap them in `Node` objects, and chain them with a lazy
`Context` that only executes when you materialize it.

## Why ThinkAgain?

- **Declarative** – Build pipelines by chaining `Node` objects in async functions
- **Transparent** – `Context` carries state and execution history you control
- **Debuggable** – Lazy execution lets you insert checks and branches naturally
- **Simple** – Just Python classes and async functions; no DSLs or hidden runtime
- **Async-first** – Native async API with a small, focused core

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

**Define your executables:**

```python
from thinkagain import Context, Executable

class Retriever(Executable):
    async def arun(self, ctx: Context) -> Context:
        # populate ctx.documents based on ctx.query
        ctx.documents = ["doc1", "doc2"]
        ctx.log("Retrieved documents")
        return ctx

class Generator(Executable):
    async def arun(self, ctx: Context) -> Context:
        ctx.answer = f"Answer based on {ctx.documents}"
        ctx.log("Generated answer")
        return ctx
```

**Declare a pipeline with Nodes:**

```python
from thinkagain import Node, run

retrieve = Node(Retriever())
generate = Node(Generator())

async def pipeline(ctx):
    ctx = await retrieve(ctx)
    ctx = await generate(ctx)
    return ctx

result = await run(pipeline, {"query": "What is ML?"})
print(result.answer)
print(result.history)
```

**Lazy materialization and control flow:**

```python
from thinkagain import LazyContext, NeedsMaterializationError

async def conditional_pipeline(ctx: LazyContext) -> LazyContext:
    ctx = await retrieve(ctx)
    ctx = await generate(ctx)

    # Materialize before branching on computed values
    ctx = await ctx  # executes pending nodes

    if ctx.answer.startswith("Answer based on"):
        ctx.log("Answer looks good")
    else:
        ctx.log("Answer needs refinement")

    return ctx
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
