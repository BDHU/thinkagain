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
   points (`run`, `Context.get`, etc.) work from both sync and async code.
4. **Observable Execution** – Every run records timing metadata and surfaces
   errors with context.
5. **Composable State** – Pipelines pass around a single `Context` object that
   holds both state and metadata.
6. **Optional Distribution** – Heavyweight services can be wrapped as replicas
   and deployed to local or remote backends without touching pipeline code.

## Execution Model

### Nodes and Pipelines

- Nodes are `async def` functions that accept and return a `Context`.
- `@node` returns a callable that simply appends itself to the context.
- `ctx = my_node(ctx)` records pending state; nothing runs yet.
- Pipelines are just Python functions that thread through contexts.

```python
@node
async def retrieve(ctx):
    ctx.set("documents", [...])
    return ctx

def pipeline(ctx):
    ctx = retrieve(ctx)       # pending
    ctx = generate(ctx)       # pending
    return ctx                # still pending
```

### Materialization Triggers

Pending nodes run when you call `run`, `arun`, `Context.materialize`,
`Context.amaterialize`, `ctx.get`, `ctx.set`, iterate, take `len`, or `await ctx`.
`peek` and `pending_names` are the two safe inspection helpers that never
materialize.

### Control Flow

Pipelines rely on plain Python flow:

- `if ctx.get("score") > 0.8:` materializes pending work before comparison.
- `while ctx.get("quality") < 0.9:` materializes once per iteration.
- Helper functions or recursion work naturally because everything passes through
  the same context object.

Inside a node you can call other nodes and `await ctx` before returning:

```python
@node
async def orchestrate(ctx):
    ctx = stage_one(ctx)
    ctx = stage_two(ctx)
    ctx = await ctx  # make sure stage_one/two ran
    return ctx
```

### Metadata & Error Handling

- Each run copies `ExecutionMetadata`, executes nodes sequentially, and records latency.
- Failures are wrapped in `NodeExecutionError` with the failing node name plus the completed ones.
- Invalid signatures trigger `NodeSignatureError` before execution.
- Returning a context with pending nodes raises `RuntimeError` so authors remember to `await ctx`.

## Distributed Replica Design

### Replica Decorator

`@replica(n=1)` turns a class into a pool of workers: it registers the spec,
adds `deploy`/`shutdown`/`get` helpers, and defers instantiation until you call
`get()` or run `distributed.deploy()`. Nodes can call replicas anywhere as long
as a backend is configured.

### Runtime Lifecycle

`distributed.runtime(...):` wraps `init → deploy → shutdown` around a block,
making sure every registered replica exists before your pipeline runs. Outside
that block, calling `ReplicaClass.get()` triggers one-off deployment, which is
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
    if ctx.get("result_count", 0) > 3:
        ctx = rerank(ctx)
    return generate(ctx)
```

3. **Self-correcting loop**

```python
def refine_until_good(ctx):
    ctx = initial(ctx)
    while ctx.get("quality", 0) < 0.85:
        ctx = refine(ctx)
        ctx = evaluate(ctx)
    return ctx
```

4. **Replica-backed node**

```python
@replica(n=2)
class Tooling:
    def invoke(self, prompt):
        ...

@node
async def call_tool(ctx):
    tool = Tooling.get()
    ctx.set("reply", tool.invoke(ctx.get("prompt")))
    return ctx
```

These patterns work identically in sync or async contexts because the only state
that matters is the `Context` object that flows through the pipeline.

## Future Work

- Additional backends (Ray, asyncio pools, etc.) can slot into the same backend
  protocol without touching the core runtime.
- ExecutionMetadata could be extended with custom hooks (e.g., tracing spans) by
  subclassing `Context` or post-processing `ctx.metadata`.
- Higher-level helpers (e.g., loop utilities) can be built as regular nodes
  while keeping the core surface minimal.
