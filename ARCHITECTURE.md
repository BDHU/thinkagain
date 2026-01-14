# ThinkAgain Architecture

ThinkAgain is intentionally small: a thin API layer feeds a minimal runtime
that schedules tasks and routes replica calls through pools/backends.

```
┌────────────┐     @node     ┌───────────────┐     .go()     ┌──────────────┐
│  async fn  │ ───────────► │ RemoteFunction │ ───────────► │ ObjectRef     │
└────────────┘               └───────────────┘              └──────────────┘
                                     │                               │
                                     │ submit Task                    │ await
                                     ▼                               ▼
                               ┌───────────────┐               ┌──────────────┐
                               │ DAGScheduler  │──────────────►│ result/error │
                               └───────────────┘               └──────────────┘

┌────────────┐    @replica    ┌───────────────┐   pool/backend ┌──────────────┐
│  class     │ ───────────► │ ActorHandle    │ ─────────────► │ Replica exec │
└────────────┘               └───────────────┘                └──────────────┘
```

## 1. API Layer (`thinkagain.api`)

- **`@node`** wraps an async function into a `RemoteFunction`. Direct calls
  run immediately; `.go()` submits a Task to the scheduler and returns an
  `ObjectRef` future.
- **`@replica`** wraps a class and returns an `ActorHandle` from `.init(...)`.
  Methods can be called with `.go()` to enqueue `ActorTask` units.

## 2. Runtime Layer (`thinkagain.runtime`)

- **`ObjectRef`** is a Ray-style future that can be awaited or passed as a
  dependency to other `.go()` calls.
- **`Task`/`ActorTask`** capture function/method calls and dependencies.
- **`DAGScheduler`** executes ready tasks concurrently and resolves dependencies.
- **Hooks** enable interception (e.g., distributed replica routing).
- **`Runtime`** provides a minimal context; if none is active, a default local
  runtime is used so `.go()` works without explicit setup.

## 3. Resources & Backends

- **Resources (`thinkagain.resources`)** define the compute topology:
  `Mesh`, `MeshNode`, and devices.
- **Backends (`thinkagain.backends`)** implement execution for replicas:
  local and gRPC by default.
- **Pools (`thinkagain.runtime.pool`)** manage replica instances and routing.

## 4. Putting It Together

1. Define nodes with `@node` and use `.go()` to build a dynamic DAG.
2. Optionally define replicas with `@replica` for stateful components.
3. Run locally with the default runtime, or enter a `Mesh` context to enable
   distributed execution and replica routing.
