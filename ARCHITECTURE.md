# ThinkAgain Architecture

ThinkAgain is intentionally small: a thin API layer feeds a minimal runtime
that schedules tasks and routes service calls through pools and backends.

```
┌────────────┐     @op       ┌───────────────┐     .go()     ┌──────────────┐
│  async fn  │ ───────────► │ RemoteFunction │ ───────────► │ ObjectRef     │
└────────────┘               └───────────────┘              └──────────────┘
                                      │                               │
                                      │ submit Task                    │ await
                                      ▼                               ▼
                                ┌───────────────┐               ┌──────────────┐
                                │ DAGScheduler  │──────────────►│ result/error │
                                └───────────────┘               └──────────────┘

┌────────────┐   @service     ┌───────────────┐   pool/backend ┌──────────────┐
│  class     │ ───────────► │ ServiceHandle  │ ─────────────► │ Service exec │
└────────────┘               └───────────────┘                └──────────────┘
```

## 1. API Layer (`thinkagain.api`)

- **`@op`** wraps an async function into a `RemoteFunction`. Direct calls
  run immediately; `.go()` submits a Task to the scheduler and returns an
  `ObjectRef` future.
- **`@service`** wraps a class and returns a `ServiceHandle` from `.init(...)`.
  Methods can be called with `.go()` to enqueue `ServiceTask` units.

## 2. Runtime Layer (`thinkagain.runtime`)

- **`ObjectRef`** is a future that can be awaited or passed as a
  dependency to other `.go()` calls.
- **`Task`/`ServiceTask`** capture function/method calls and dependencies.
- **`DAGScheduler`** executes ready tasks concurrently and resolves dependencies.
- **Hooks** enable interception (e.g., distributed service routing).
- **`Runtime`** provides a minimal context; if none is active, a default local
  runtime is used so `.go()` works without explicit setup.

## 3. Resources & Backends

- **Resources (`thinkagain.resources`)** define the compute topology:
  `Mesh`, `MeshNode`, and devices.
- **Backends (`thinkagain.backends`)** implement execution for services:
  local and gRPC by default.
- **Pools (`thinkagain.runtime.pool`)** manage service instances and routing.

## 4. Dynamic Execution Model

Unlike static compilation approaches, ThinkAgain builds computation graphs
dynamically at runtime:

1. **Task Submission**: When you call `fn.go()`, the system immediately
   returns an `ObjectRef` without waiting for execution
2. **Dependency Tracking**: `ObjectRef`s can be passed as arguments to other
   `.go()` calls, automatically building the dependency graph
3. **Scheduler Optimization**: The `DAGScheduler` finds independent tasks
   and executes them in parallel
4. **Lazy Evaluation**: Actual computation only happens when you `await`
   an `ObjectRef`

## 5. Service Architecture

Services provide stateful, distributed execution:

- **Instance Management**: Each service maintains its own state across calls
- **Resource Allocation**: Services can request specific resources (GPUs, CPUs)
- **Backend Abstraction**: Services run locally or remotely via gRPC without code changes
- **Scaling**: Multiple instances can be created for load balancing

## 6. Putting It Together

1. Define ops with `@op` and use `.go()` to build a dynamic DAG.
2. Optionally define services with `@service` for stateful components.
3. Run locally with the default runtime, or enter a `Mesh` context to enable
   distributed execution and service routing.
4. The scheduler automatically parallelizes independent tasks while respecting
   dependencies.
