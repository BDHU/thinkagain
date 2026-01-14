# ThinkAgain

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/BDHU/thinkagain?utm_source=badge)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A minimal framework for building distributed AI pipelines. Write async functions, compose them with `.go()`, and scale them across clusters.

## Features

- **Dynamic execution** – Submit tasks with `.go()` and let the scheduler handle parallelism automatically
- **Transparent distribution** – `@service` marks classes for distributed execution
- **Automatic parallelism** – Independent tasks run in parallel without extra code
- **Built-in profiling** – Automatic dependency tracking and performance metrics
- **Minimal** – Clean API with no complex schedulers or DSLs

## Install

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

### Basic Usage

```python
import thinkagain as ta

@ta.op
async def add_one(x: int) -> int:
    return x + 1

@ta.op
async def multiply(x: int, factor: int) -> int:
    return x * factor

async def pipeline(x: int) -> int:
    # Submit tasks immediately (non-blocking)
    x_ref = add_one.go(x)
    result_ref = multiply.go(x_ref, 3)
    
    # Wait for result
    return await result_ref

result = await pipeline(5)  # Returns 18
```

### Distributed Execution

```python
import thinkagain as ta

# CPU-only service
@ta.service()
class Retriever:
    async def retrieve(self, query: str) -> list[str]:
        return ["doc1", "doc2", "doc3"]

# GPU service
@ta.service(gpus=4)
class Generator:
    def __init__(self):
        self.model = load_llm()

    async def generate(self, query: str, docs: list[str]) -> str:
        return f"Answer based on {len(docs)} documents"

# Define cluster resources
mesh = ta.Mesh([
    ta.MeshNode("server1", gpus=8, cpus=32),
    ta.MeshNode("server2", gpus=8, cpus=32),
])

# Create service handles
retriever = Retriever.init()
generator = Generator.init()

async def rag_pipeline(query: str) -> str:
    docs_ref = retriever.retrieve.go(query)
    return await generator.generate.go(query, docs_ref)

# Execute with mesh
with mesh:
    result = await rag_pipeline("What is ML?")
    print(result)
```

## Core Concepts

- **`@op`** – Decorator for async functions
- **`@service`** – Decorator for distributed classes
- **`.go()`** – Submit work and return ObjectRef future
- **`Mesh`** – Define cluster resources (GPUs, CPUs)
- **`ObjectRef`** – Future that can be awaited or passed to other calls

## API Overview

### Core Decorators
- `@ta.op` – Mark async functions for dynamic composition
- `@ta.service(gpus=None)` – Mark classes for distributed execution

### Execution
- `fn.go(*args)` – Submit call and return ObjectRef
- `await object_ref` – Wait for result

### Resources
- `ta.Mesh(devices)` – Define cluster topology
- `ta.GpuDevice(id)` – GPU device
- `ta.CpuDevice(count)` – CPU resources
- `ta.devices()` – Auto-detect GPUs

### Profiling
- `ta.profile()` – Context manager for profiling
- `ta.get_profiler()` – Access profiler instance

## Examples

See [examples/](examples/) for working demos:
- `demo.py` – Core API with dynamic execution
- `distributed_example.py` – Distributed execution with services

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run examples
uv run python examples/demo.py
```

## License

Apache 2.0 – see [LICENSE](LICENSE)
