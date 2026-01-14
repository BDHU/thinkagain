# ThinkAgain Design

This document captures the principles behind ThinkAgain's current
implementation: keep the runtime tiny, make dynamic execution obvious, and
provide an easy escape hatch for stateful services without building a
workflow engine.

## Design Goals

1. **Plain Python Pipelines** – Users write regular functions with familiar
   control flow (`if`, `for`, `while`). No DSL, no custom graph builder.
2. **Dynamic by Default** – The framework builds computation graphs at runtime
   as `.go()` calls are made. This enables natural parallelism and makes
   debugging easier.
3. **Async-First but Sync-Friendly** – Ops are async functions, yet the entry
   points work from both sync and async code.
4. **DAG Execution** – The scheduler tracks dependencies between ObjectRefs,
   enabling fanout patterns where shared ancestors execute only once.
5. **Composable State** – Services maintain their own state across multiple
   method calls, enabling patterns like caching and session management.
6. **Optional Distribution** – Heavyweight services can be wrapped as services
   and deployed to local or remote backends without touching pipeline code.

## Execution Model

### Ops and Dynamic Graphs

- Ops are `async def` functions that accept and return plain Python values.
- `@op` wraps the function so it can be submitted via `.go()` calls.
- `ref = my_op.go(args)` immediately returns an `ObjectRef` future; nothing runs yet.
- `ObjectRef`s can be passed as arguments to other `.go()` calls, building
  the dependency graph automatically.
- Pipelines are just Python functions that orchestrate `.go()` calls.

```python
@op
async def retrieve(query: str) -> list[str]:
    return ["doc1", "doc2", "doc3"]

@op
async def generate(docs: list[str], query: str) -> str:
    return f"Answer for '{query}' using {len(docs)} docs"

async def rag_pipeline(query: str) -> str:
    # Submit tasks immediately (non-blocking)
    docs_ref = retrieve.go(query)
    answer_ref = generate.go(docs_ref, query)
    
    # Wait for final result
    return await answer_ref
```

### Execution Triggers

Pending tasks run when you `await` an `ObjectRef`. The `DAGScheduler`
walks the dependency graph to build topological order, deduplicating
shared ancestors and executing independent tasks in parallel.

### Control Flow

Pipelines rely on plain Python flow:

- `if (await ref).score > 0.8:` materializes pending work before comparison.
- `while (await ref).quality < 0.9:` materializes once per iteration.
- Helper functions or recursion work naturally because everything passes through
  `ObjectRef`s.

Inside an op you can call other ops and `await` the result:

```python
@op
async def orchestrate(query: str) -> str:
    docs_ref = retrieve.go(query)
    answer_ref = generate.go(docs_ref, query)
    
    # Wait for intermediate result if needed
    docs = await docs_ref
    if len(docs) > 5:
        # Submit different work based on intermediate result
        return await summarize.go(answer_ref)
    
    return await answer_ref
```

### Fanout and Shared Dependencies

Multiple operations can share the same `ObjectRef` dependency:

```python
@op
async def process_a(data: str) -> str:
    return f"Processed A: {data}"

@op
async def process_b(data: str) -> str:
    return f"Processed B: {data}"

@op
async def combine(a: str, b: str) -> str:
    return f"{a}\n{b}"

async def fanout_pipeline(input_data: str) -> str:
    # All branches share the same base_ref - it executes once
    base_ref = fetch_data.go(input_data)
    
    # Parallel fanout
    ref_a = process_a.go(base_ref)
    ref_b = process_b.go(base_ref)
    
    # Fanin
    return await combine.go(ref_a, ref_b)
```

### Error Handling

- Failures are wrapped in `OpExecutionError` with the failing op name.
- Returning an `ObjectRef` without awaiting it works fine for pipeline
  construction, but individual results must be awaited for values.

**Concurrency constraint:** Concurrent execution of tasks that share
dependencies is automatically handled by the scheduler, but manual `asyncio.gather()`
on tasks with shared dependencies should be done carefully.

## Service Design

### Service Decorator

`@service(gpus=0)` turns a class into a distributed service:
- Returns a `ServiceHandle` from `.init()` that exposes methods
- Methods can be called with `.go()` to submit work to the service instance
- Services maintain mutable state across multiple method calls
- Multiple instances can be created for load balancing

### Service Lifecycle

Services have a clear lifecycle:
1. **Definition**: Define class with `@service(...)`
2. **Handle Creation**: `handle = MyService.init()` returns a handle
3. **Deployment**: Services deploy when entering a `Mesh` context or on first use
4. **Execution**: Method calls via `handle.method.go(args)` submit work
5. **Cleanup**: Services shutdown when exiting mesh context

### Backends and Remote Execution

- **Local backend**: Default, keeps service instances in the current process.
- **gRPC backend**: Returns proxies whose method calls serialize to remote servers.
- Switching backends is a configuration change, not a code change.

## Common Patterns

### 1. Linear Pipeline

```python
async def rag(query: str) -> str:
    docs_ref = retrieve.go(query)
    context_ref = combine_docs.go(docs_ref)
    answer_ref = generate.go(context_ref, query)
    return await answer_ref
```

### 2. Conditional Pipeline

```python
async def maybe_rerank(query: str) -> str:
    docs_ref = retrieve.go(query)
    docs = await docs_ref
    
    if len(docs) > 3:
        docs_ref = rerank.go(query, docs_ref)
    
    answer_ref = generate.go(docs_ref, query)
    return await answer_ref
```

### 3. Self-Correcting Loop

```python
async def refine_until_good(query: str) -> str:
    answer_ref = generate.go(query)
    quality_ref = evaluate.go(answer_ref)
    quality = await quality_ref
    
    while quality < 0.85:
        refined_ref = refine.go(answer_ref, quality_ref)
        answer_ref = generate.go(query, refined_ref)
        quality_ref = evaluate.go(answer_ref)
        quality = await quality_ref
    
    return await answer_ref
```

### 4. Fanout with Shared Ancestor

```python
@op
async def merge(results: list[str]) -> str:
    return "\n".join(results)

async def fanout_pipeline(query: str) -> str:
    base_ref = fetch_data.go(query)  # Shared ancestor
    
    # Multiple consumers of same base_ref - executes once
    left_ref = process_a.go(base_ref)
    right_ref = process_b.go(base_ref)
    
    # Fanin
    merged_ref = merge.go([left_ref, right_ref])
    return await merged_ref
```

### 5. Service-Backed Pipeline

```python
@service()
class Cache:
    def __init__(self):
        self.cache = {}
    
    async def get(self, key: str) -> str | None:
        return self.cache.get(key)
    
    async def put(self, key: str, value: str):
        self.cache[key] = value

@op
async def cached_retrieve(cache: Cache, query: str) -> list[str]:
    cached_ref = cache.get.go(query)
    cached = await cached_ref
    
    if cached is not None:
        return cached
    
    docs_ref = retrieve.go(query)
    docs = await docs_ref
    
    # Cache the result (state persists in service)
    await cache.put.go(query, docs)
    return docs

async def pipeline(query: str) -> str:
    cache = Cache.init()
    docs_ref = cached_retrieve.go(cache, query)
    answer_ref = generate.go(docs_ref, query)
    return await answer_ref
```

These patterns work identically in local or distributed contexts because
the only state that matters is the `ObjectRef`s and service instances that
flow through the pipeline.

## Dynamic Execution Benefits

The dynamic execution model provides several advantages:

1. **Natural Parallelism**: Submit multiple tasks and let the scheduler
   automatically parallelize independent work
2. **Flexible Control Flow**: Use regular Python control flow with
   intermediate results to decide next steps
3. **Incremental Construction**: Build pipelines incrementally without
   upfront graph definition
4. **Debugging**: Debug individual ops in isolation before composing
5. **Resource Awareness**: The scheduler understands resource requirements
   and schedules accordingly

## Future Work

- Additional backends (cluster managers, cloud services) can slot into the
  same backend protocol without touching core runtime
- Advanced scheduling strategies (priority, fairness) can be added to DAGScheduler
- Observability hooks (metrics, tracing) can be added via op wrappers
- Higher-level helpers (workflow templates) can be built as regular ops
  while keeping core surface minimal
