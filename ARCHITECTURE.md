# ThinkAgain Architecture

ThinkAgain is intentionally small: the entire framework is organized around a
lazy `Context`, a thin `Node` wrapper, and a lightweight distributed runtime
for stateful services. This document explains how those pieces fit together.

```
┌────────────┐       ┌──────────┐       ┌────────────────────────┐
│  Node fn   │  @node│  Node    │  >>   │ Context (pending state)│
└────────────┘       └──────────┘       └────────────────────────┘
        │                          │
        │ run/arun                 │
        ▼                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Lazy materialization + ExecutionMetadata + error handling      │
└────────────────────────────────────────────────────────────────┘
        │                          │
        │ needs stateful worker    │
        ▼                          ▼
┌────────────┐ deploy  ┌──────────────┐ get   ┌──────────────────┐
│ @replica   │ ──────► │ ReplicaSpec  │ ────► │ Backend (local / │
│ classes    │         └──────────────┘       │ gRPC proxy)      │
└────────────┘                                 └──────────────────┘
```

## 1. Core Runtime

### Context

`thinkagain.core.context.Context` is a dict-like state container with three
slots: `_data` (mutable state), `_pending` (nodes queued for execution), and
`_metadata` (`ExecutionMetadata`). Methods such as `get`, `set`, `items`,
iteration, and `len` materialize pending nodes before touching `_data`, while
`peek` lets you inspect values without running anything. Metadata holds
timestamps, per-node latency totals, and execution counts; it is copied once
when a run starts so chaining stays cheap.

### Node and `@node`

`thinkagain.core.node.Node` wraps an `async def (ctx) -> ctx`. On first
execution it validates the signature, then simply appends itself to the
context's pending list when called. `execute()` awaits the user function and
returns the resulting context. The `@node` decorator is only syntactic sugar.

### Materialization

Nothing runs until you call `run`, `arun`, `Context.materialize`,
`Context.amaterialize`, or a method that needs actual data (`get`, iteration,
`len`, etc.). During `_run_pending_async` the metadata copy is made once, each
node executes sequentially, and errors are wrapped in `NodeExecutionError`
containing the failing node plus the ones that finished beforehand. Returning a
context that still has pending nodes raises a `RuntimeError`, which nudges node
authors to `await ctx` before returning.

## 2. Distributed Replica Runtime

Some pipelines require stateful helpers such as LLM client pools, vector stores,
or tool adapters. ThinkAgain bundles a small replica runtime for those cases.

### ReplicaSpec and `@replica`

Decorating a class with `@thinkagain.distributed.replica` registers a
`ReplicaSpec` (class reference, pool size, cached constructor arguments). The
decorator adds `deploy`, `shutdown`, and `get` classmethods. Specs live in a
registry so the runtime can deploy or tear everything down at once.

### Runtime Configuration

`thinkagain.distributed.runtime` stores the backend choice (`local` or `grpc`),
the server address, and options. `get_backend()` lazily instantiates that backend,
`reset_backend()` clears it, and the `runtime(...)` context manager wraps
`init → deploy → shutdown` around a code block. The pipeline runtime never needs
to know where replicas live.

### Backends

Two backends implement the shared protocol:

- **LocalBackend** – Keeps replicas in-process and rotates through them.
- **GrpcBackend** – Connects to the bundled gRPC server, requests deploy/shutdown,
  and returns proxies that pickle method calls.

## 3. Putting It Together

1. **Define nodes** with `@node` and write plain Python pipelines that receive
   a `Context`. Pipelines can use any control flow (`if`, `while`, recursion)
   because nodes only run when the context is materialized.
2. **Optionally declare replicas** for heavy, stateful resources and wrap the
   code that runs nodes inside `distributed.runtime(...)`.
3. **Execute** via `run`/`arun` and inspect `ctx.metadata` (latencies) or
   `ctx.to_dict()` for the final state.

Because there is only one execution path and one state container, the runtime
stays predictable and easy to debug while still supporting scaling out stateful
services when needed.
