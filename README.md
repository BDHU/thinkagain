# ThinkAgain

[![PyPI version](https://img.shields.io/pypi/v/thinkagain.svg)](https://pypi.org/project/thinkagain/)
[![License](https://img.shields.io/pypi/l/thinkagain.svg)](https://github.com/BDHU/thinkagain/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BDHU/thinkagain)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/BDHU/thinkagain/test.yml?branch=main)](https://github.com/BDHU/thinkagain/actions/workflows/test.yml)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/BDHU/thinkagain?utm_source=badge)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A minimal framework for building distributed AI pipelines with JAX-style graph compilation. Write async functions, compose them with `@jit`, and scale them across clusters with `@replica` – all with transparent profiling and optimization.

## Why ThinkAgain?

- **JAX-inspired API** – Familiar `@jit` compilation with `cond`, `while_loop`, and `scan` for control flow
- **Graph compilation** – Functions are traced into static computation graphs for efficient execution
- **Transparent distribution** – `@replica` marks classes for distributed execution, `Mesh` defines resources
- **Clean separation** – Single `@jit` decorator for local and distributed, mesh determines execution mode
- **Built-in profiling** – Automatic dependency tracking and performance metrics
- **Minimal** – ~500 LOC core, no complex schedulers or DSLs

## Core Concepts

- **`@node`** – Decorator for async functions that can be composed in graphs
- **`@jit`** – JAX-style tracing that compiles async functions into static graphs
- **`@replica`** – Decorator for stateful classes that can be distributed across multiple instances
- **`Mesh`** – Explicit resource topology (GPUs, CPUs, nodes) for distributed execution
- **`cond`, `while_loop`, `scan`, `switch`** – Control flow primitives for traced functions

## Installation

```bash
pip install thinkagain
# or with uv
uv add thinkagain
```

## Quick Start

### Local Execution with @jit

```python
import thinkagain as ta

@ta.node
async def add_one(x: int) -> int:
    return x + 1

@ta.node
async def multiply(x: int, factor: int) -> int:
    return x * factor

@ta.jit
async def pipeline(x: int) -> int:
    x = await add_one(x)
    x = await multiply(x, 3)
    return x

result = await pipeline(5)  # Returns 18
```

### Control Flow with cond and while_loop

```python
import thinkagain as ta

@ta.node
async def increment(x: int) -> int:
    return x + 1

@ta.node
async def decrement(x: int) -> int:
    return x - 1

@ta.jit
async def pipeline(x: int, target: int) -> int:
    # Loop until we reach target
    x = await ta.while_loop(
        lambda s: s < target,
        increment,
        x
    )

    # Conditional branching
    x = await ta.cond(
        lambda s: s % 2 == 0,
        increment,
        decrement,
        x
    )

    return x

result = await pipeline(0, 10)
```

## Distributed Execution

### Define a Mesh and Use Replicas

```python
import thinkagain as ta

# Define your cluster topology
mesh = ta.Mesh([
    ta.MeshNode("server1", gpus=8, cpus=32),
    ta.MeshNode("server2", gpus=8, cpus=32),
])

# CPU-only replica
@ta.replica()
class Retriever:
    def __init__(self):
        pass

    async def retrieve(self, query: str) -> list[str]:
        # Retrieval logic - scales freely on CPU
        return ["doc1", "doc2", "doc3"]

# GPU-accelerated replica
@ta.replica(gpus=4)  # 4 GPUs per instance
class Generator:
    def __init__(self):
        self.model = load_llm()

    async def generate(self, query: str, docs: list[str]) -> str:
        # LLM generation - requires GPUs
        return f"Answer based on {len(docs)} documents"

# Create handles outside @jit
retriever = Retriever.init()
generator = Generator.init()

@ta.jit
async def rag_pipeline(query: str) -> str:
    docs = await retriever.retrieve(query)
    return await generator.generate(query, docs)

# Execute with mesh context
with mesh:
    result = await rag_pipeline("What is ML?")
    print(result)
```

### Stateful Replicas with Setup

```python
import thinkagain as ta

def setup_model():
    """Called once per instance to initialize state."""
    return {"model": load_llm(), "cache": {}}

@ta.replica(gpus=2, setup=setup_model)
class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def inference(self, prompt: str) -> str:
        """State from setup is available in actual execution."""
        return f"Generated response for {prompt}"

# Create handle
llm = LLM.init("llama-70b")

@ta.jit
async def pipeline(prompt: str) -> str:
    return await llm.inference(prompt)

mesh = ta.Mesh([ta.GpuDevice(0), ta.GpuDevice(1)])

with mesh:
    result = await pipeline("Hello world")
```

## Transparent Profiling

Profiling runs automatically in the background and can be queried at any time:

```python
import thinkagain as ta

@ta.replica()
class Processor:
    def __init__(self):
        pass

    async def process(self, x: int) -> int:
        return x * 2

processor = Processor.init()

@ta.jit
async def pipeline(x: int) -> int:
    return await processor.process(x)

mesh = ta.Mesh([ta.CpuDevice(4)])

with ta.profile() as profiler:
    with mesh:
        for i in range(100):
            await pipeline(i)

    # Get profiling data
    summary = profiler.summary()
    print(summary['dependency_graph'])  # Which functions call which
    print(summary['fanout_matrix'])     # Call patterns
    print(summary['execution_stats'])   # Timing stats
```

## Features

### Multi-way Branching with switch

```python
import thinkagain as ta

@ta.node
async def route_a(x: int) -> int:
    return x * 2

@ta.node
async def route_b(x: int) -> int:
    return x * 3

@ta.node
async def route_c(x: int) -> int:
    return x * 4

@ta.jit
async def pipeline(x: int, choice: int) -> int:
    return await ta.switch(
        lambda _: choice,
        [route_a, route_b, route_c],
        x
    )
```

### Scans for Sequential Processing

```python
import thinkagain as ta

@ta.node
async def accumulate(carry: int, x: int) -> tuple[int, int]:
    new_carry = carry + x
    output = new_carry
    return new_carry, output

@ta.jit
async def sum_sequence(init: int, xs: list[int]) -> tuple[int, list[int]]:
    # Scan applies accumulate to each element
    return await ta.scan(accumulate, init, xs)

final_carry, outputs = await sum_sequence(0, [1, 2, 3, 4, 5])
# final_carry = 15, outputs = [1, 3, 6, 10, 15]
```

### Auto-detect Available Resources

```python
import thinkagain as ta

# Auto-detect GPUs
gpus = ta.devices()  # Returns list of available GPUs
mesh = ta.Mesh(gpus)

# Auto-detect CPUs
cpu = ta.cpus()  # Returns CPU device with core count
mesh = ta.Mesh([cpu])
```

## Design Philosophy

ThinkAgain follows JAX's philosophy of explicit resource management and transparent compilation:

1. **Explicit is better than implicit** – You define the mesh, mark classes for replication
2. **Compilation separates concerns** – Write local logic, compile to distributed execution
3. **Tracing enables optimization** – Static graphs allow for profiling and optimization
4. **Minimal magic** – No auto-scaling, no hidden deployments, everything is explicit

## API Summary

### Core Decorators
- `@ta.node` – Mark async functions for graph composition
- `@ta.jit` – Compile functions into static computation graphs
- `@ta.replica(gpus=None, setup=None)` – Mark classes for distributed execution

### Control Flow
- `ta.cond(pred_fn, true_fn, false_fn, operand)` – Conditional branching
- `ta.while_loop(cond_fn, body_fn, init)` – Loops
- `ta.scan(body_fn, init, xs)` – Sequential processing
- `ta.switch(index_fn, branches, operand)` – Multi-way branching

### Resource Management
- `ta.Mesh(devices)` – Define cluster topology
- `ta.GpuDevice(id)` – GPU device
- `ta.CpuDevice(count)` – CPU resources
- `ta.MeshNode(name, gpus=0, cpus=0)` – Multi-GPU/CPU node
- `ta.devices()` – Auto-detect GPUs

### Profiling
- `ta.profile()` – Context manager for profiling
- `ta.enable_profiling()` / `ta.disable_profiling()` – Manual control
- `ta.get_profiler()` – Access profiler instance

## Examples

See [examples/](examples/) for working demos:
- `demo.py` – Core @jit API with control flow (cond, while_loop, scan)
- `new_distributed_api_demo.py` – Distributed execution with @replica and Mesh
- `grpc_remote_example.py` – Remote execution with gRPC backend (multi-server)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) – System architecture and design decisions
- [DESIGN.md](DESIGN.md) – Execution model and control flow
- [examples/](examples/) – Working examples and patterns

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run examples
uv run python examples/new_distributed_api_demo.py
```

## License

Apache 2.0 – see [LICENSE](LICENSE)
