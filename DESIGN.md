# ThinkAgain Design

This document captures the principles behind ThinkAgain's current
implementation: keep the runtime tiny, make lazy execution obvious, and
provide an easy escape hatch for stateful services without building a
workflow engine.

## Design Goals

1. **Plain Python Pipelines** – Users write regular functions with familiar
   control flow (`if`, `for`, `while`). No DSL, no custom graph builder.
2. **Lazy by Default** – The framework collects pending node calls and only
   executes when results are needed. This enables conditional logic that depends
   on partial results and makes debugging easier.
3. **Async-First but Sync-Friendly** – Nodes are async functions, yet the entry
   points (`run`, `ctx.data`, etc.) work from both sync and async code.
4. **DAG Execution** – Contexts track dependencies via back-pointers, enabling
   fanout patterns where shared ancestors execute only once.
5. **Composable State** – Pipelines pass around a single `Context` object that
   wraps user data (dataclasses, dicts, etc.).
6. **Optional Distribution** – Heavyweight services can be wrapped as replicas
   and deployed to local or remote backends without touching pipeline code.

## Execution Model

### Nodes and Pipelines

- Nodes are `async def` functions that accept and return plain Python values.
- `@node` wraps the function so it creates a new `Context` with back-pointers.
- `ctx = my_node(ctx)` records a pending node with dependency tracking; nothing runs yet.
- Multi-input nodes are supported: each `Context` argument becomes a tracked parent.
- Pipelines are just Python functions that thread through contexts.

```python
from dataclasses import dataclass

@dataclass
class State:
    query: str
    documents: list[str] | None = None

@node
async def retrieve(s: State) -> State:
    return State(query=s.query, documents=["doc1", "doc2"])

def pipeline(ctx):
    ctx = retrieve(ctx)       # pending
    ctx = generate(ctx)       # pending
    return ctx                # still pending
```

### Materialization Triggers

Pending nodes run when you call `run`, `arun`, access `ctx.data`, or `await ctx`.
The `DAGExecutor` walks back-pointers to build topological order, deduplicating
shared ancestors.

### Control Flow

Pipelines rely on plain Python flow:

- `if ctx.data.score > 0.8:` materializes pending work before comparison.
- `while ctx.data.quality < 0.9:` materializes once per iteration.
- Helper functions or recursion work naturally because everything passes through
  the same context object.

Inside a node you can call other nodes and `await ctx` before returning:

```python
@node
async def orchestrate(ctx):
    ctx = stage_one(ctx)
    ctx = stage_two(ctx)
    value = await ctx  # make sure stage_one/two ran
    return value
```

### Fanout and Multi-Input Nodes

Nodes can accept multiple context arguments. Shared ancestors execute only once:

```python
@node
async def combine(a: int, b: int) -> int:
    return a + b

# Both branches share `base`; it runs only once
base = some_node(ctx)
left = process_a(base)
right = process_b(base)
result = combine(left, right)
```

### Error Handling

- Failures are wrapped in `NodeExecutionError` with the failing node name.
- Returning a context with pending nodes raises `RuntimeError` so authors remember to `await ctx`.

**Concurrency constraint:** Concurrent materialization of contexts that share
ancestors (e.g., via `asyncio.gather()`) is not supported. Materialize sequentially.

## Distributed Replica Design

### Replica Decorator

`@replica(n=1)` turns a class into a pool of workers: it registers the spec with
the default `ReplicaManager` (or an explicit manager) and returns a
`ReplicaHandle` that exposes `deploy`/`shutdown`/`get`. Instantiation is
deferred until you call `get()` or run `manager.deploy_all()`. Nodes can call
replicas anywhere as long as a backend is configured.

### Runtime Lifecycle

`distributed.runtime(...):` wraps `init → deploy → shutdown` around a block,
making sure every registered replica exists before your pipeline runs. Outside
that block, calling `ReplicaHandle.get()` triggers one-off deployment, which is
handy for quick tests.

### Backends and Remote Execution

- **Local backend**: default, keeps replica instances in the current process.
- **gRPC backend**: returns proxies whose attribute access serializes calls to a
  remote server. Switching to gRPC is a configuration change.

## Common Patterns

1. **Linear pipeline**

```python
def rag(ctx):
    return generate(rerank(retrieve(ctx)))
```

2. **Conditional routing**

```python
def maybe_rerank(ctx):
    ctx = retrieve(ctx)
    if len(ctx.data.documents) > 3:
        ctx = rerank(ctx)
    return generate(ctx)
```

3. **Self-correcting loop**

```python
def refine_until_good(ctx):
    ctx = initial(ctx)
    while ctx.data.quality < 0.85:
        ctx = refine(ctx)
        ctx = evaluate(ctx)
    return ctx
```

4. **Fanout with shared ancestor**

```python
@node
async def merge(a: State, b: State) -> State:
    return State(results=a.results + b.results)

def fanout_pipeline(ctx):
    base = fetch_data(ctx)
    left = process_a(base)   # shares base
    right = process_b(base)  # shares base
    return merge(left, right)  # base runs once
```

5. **Replica-backed node**

```python
@replica(n=2)
class Tooling:
    def invoke(self, prompt):
        ...

@node
async def call_tool(s: State) -> State:
    tool = Tooling.get()
    return State(reply=tool.invoke(s.prompt))
```

These patterns work identically in sync or async contexts because the only state
that matters is the `Context` object that flows through the pipeline.

## Future Work

- Additional backends (Ray, asyncio pools, etc.) can slot into the same backend
  protocol without touching the core runtime.
- Observability hooks (tracing, metrics) could be added via node wrappers or
  custom execution callbacks.
- Higher-level helpers (e.g., loop utilities) can be built as regular nodes
  while keeping the core surface minimal.
