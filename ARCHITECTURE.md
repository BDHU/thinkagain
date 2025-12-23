# ThinkAgain Architecture

ThinkAgain is intentionally small: the entire framework is organized around a
lazy `Context`, a thin `Node` wrapper, and a lightweight distributed runtime
for stateful services. This document explains how those pieces fit together.

```
┌────────────┐       ┌──────────┐       ┌─────────────────────────┐
│  Node fn   │  @node│  Node    │  >>   │ Context (back-pointers) │
└────────────┘       └──────────┘       └─────────────────────────┘
        │                          │
        │ run/arun                 │
        ▼                          ▼
┌────────────────────────────────────────────────────────────────┐
│ DAG traversal + deduplication + error handling                  │
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

`thinkagain.core.context.Context` wraps user data and tracks execution
dependencies via back-pointers. Key slots:

- `_data` – The user's state (dataclass, dict, or any Python value).
- `_parents` – Tuple of parent `Context` objects this context depends on.
- `_call_args` / `_call_kwargs` – Arguments passed when the node was called.
- `_node` – The node that will produce this context's value.
- `_executed` – Flag indicating whether the node has run.

This design enables DAG execution: when multiple branches share a common
ancestor, that ancestor executes only once. Access `ctx.data` to get the
underlying value (triggering materialization if pending).

### Node and `@node`

`thinkagain.core.node.Node` wraps an async function that accepts one or more
arguments and returns a value. When called, it creates a new `Context` with
back-pointers to all input contexts rather than executing immediately.
Multi-input nodes are supported: each `Context` argument becomes a tracked
parent. The `@node` decorator is syntactic sugar for `FunctionNode`.

### Materialization

Nothing runs until you call `run`, `arun`, `Context.materialize`,
`Context.amaterialize`, or access `ctx.data`. The `DAGExecutor` walks the
back-pointer graph via `traverse_pending()`, building a topological execution
order. Each pending context executes once; shared ancestors are deduplicated
automatically. Errors are wrapped in `NodeExecutionError` containing the
failing node name.

**Note:** Concurrent materialization of contexts that share ancestors (e.g.,
via `asyncio.gather()`) is not supported. Materialize sequentially.

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
   a `Context`. Nodes can accept multiple inputs for fanout patterns. Pipelines
   can use any control flow (`if`, `while`, recursion) because nodes only run
   when the context is materialized.
2. **Optionally declare replicas** for heavy, stateful resources and wrap the
   code that runs nodes inside `distributed.runtime(...)`.
3. **Execute** via `run`/`arun` and access `result.data` for the final state.

Because contexts track dependencies via back-pointers and execute as a DAG, the
runtime stays predictable and easy to debug while enabling fanout patterns
where shared ancestors run only once.
