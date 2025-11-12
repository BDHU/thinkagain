# thinkagain Architecture

## Philosophy: Everything is a Graph

The core insight of thinkagain is that **all execution can be modeled as graphs**, and **graphs compose naturally**.

Instead of having separate abstractions for pipelines, workflows, agents, and subgraphs, we have one unified concept: **Executable**.

## Core Hierarchy

```
Executable (base interface)
├── Worker (leaf computations - your business logic)
├── Graph (DAG with cycles and conditional routing)
└── Pipeline (syntactic sugar for sequential graphs)
```

All three implement the same interface:
- `__call__(ctx) -> ctx` - synchronous execution
- `arun(ctx) -> ctx` - asynchronous execution
- `__rshift__(other)` - composition via `>>` operator

## Key Design Decisions

### 1. Unified Interface

Everything that transforms Context is an `Executable`. This means:

```python
# These all work the same way
worker = MyWorker()
graph = Graph(...)
pipeline = Pipeline(...)

# All can be called
result = worker(ctx)
result = graph(ctx)
result = pipeline(ctx)

# All can be composed
flow = worker >> graph >> pipeline
```

### 2. Async-First Execution

The framework is async-first but supports sync:
- Primary execution: `await graph.arun(ctx)`
- Sync wrapper: `graph(ctx)` calls `asyncio.run(arun(ctx))`
- Sync workers automatically wrapped in executors

This simplifies the codebase significantly - only one execution path to maintain.

### 3. Pipeline = Graph

`Pipeline` is not a separate abstraction - it's literally a `Graph` that auto-wires nodes sequentially:

```python
class Pipeline(Graph):
    def __init__(self, nodes):
        super().__init__()
        # Auto-wire nodes in sequence
        for i, node in enumerate(nodes):
            self.add_node(f"_{i}", node)
            if i > 0:
                self.add_edge(f"_{i-1}", f"_{i}")
```

This means pipelines get all graph features for free: visualization, introspection, etc.

### 4. Natural Subgraph Composition

Because everything is an `Executable`, subgraphs "just work":

```python
# Create subgraphs
research_agent = Graph(name="research")
research_agent.add_node("query", QueryWorker())
research_agent.add_node("search", SearchWorker())
# ... configure ...

writing_agent = Graph(name="writing")
# ... configure ...

# Compose them in a coordinator graph
coordinator = Graph(name="coordinator")
coordinator.add_node("research", research_agent)  # Graph as node!
coordinator.add_node("write", writing_agent)      # Graph as node!
coordinator.add_edge("research", "write")
```

**No special API needed.** Graphs are just executables, so they can be nodes in other graphs.

### 5. Eliminated Features

To stay minimal, we removed:
- **Conditional/Switch/Loop control flow** - Use Graph with conditional edges instead
- **Step-by-step debugging** - Simplified to logs and visualization
- **Parallel execution** - Can be added back as a separate Worker if needed
- **Separate sync/async paths** - One async path, sync wraps it

## Code Metrics

### Before (v0.1.0)
- `graph.py`: 676 lines
- `pipeline.py`: 505 lines
- `worker.py`: 142 lines
- **Total core**: ~1,566 lines

### After (v0.1.1)
- `executable.py`: 151 lines (new base)
- `graph.py`: 472 lines (-204)
- `pipeline.py`: 81 lines (-424)
- `worker.py`: 144 lines (+2)
- **Total core**: ~848 lines

**~46% code reduction** while gaining subgraph composition!

## Usage Patterns

### Pattern 1: Simple Sequential Pipeline

```python
# Option A: Using >> operator (most concise)
pipeline = worker1 >> worker2 >> worker3
result = await pipeline.arun(ctx)

# Option B: Explicit Pipeline
pipeline = Pipeline([worker1, worker2, worker3])
result = await pipeline.arun(ctx)
```

### Pattern 2: Graph with Cycles

```python
graph = Graph(name="self_correcting")
graph.add_node("retrieve", retriever)
graph.add_node("critique", critic)
graph.add_node("refine", refiner)

graph.add_edge("retrieve", "critique")
graph.add_conditional_edge(
    "critique",
    route=lambda ctx: "done" if ctx.quality > 0.8 else "refine",
    paths={"done": END, "refine": "refine"}
)
graph.add_edge("refine", "retrieve")  # Cycle!

result = await graph.arun(ctx)
```

### Pattern 3: Multi-Agent with Subgraphs

```python
# Build specialized agents
agent1 = Graph(name="specialist1")
# ... configure agent1 ...

agent2 = Graph(name="specialist2")
# ... configure agent2 ...

# Orchestrate in coordinator
coordinator = Graph(name="coordinator")
coordinator.add_node("agent1", agent1)  # Subgraph!
coordinator.add_node("agent2", agent2)  # Subgraph!
coordinator.add_edge("agent1", "agent2")

# Or use >> for linear composition
system = agent1 >> agent2
```

### Pattern 4: State Transformation Between Subgraphs

```python
# Subgraph with different context needs
def transform_context(ctx: Context) -> Context:
    """Adapter for subgraph with different schema."""
    sub_ctx = Context(specialized_data=ctx.general_data)
    result_ctx = subgraph(sub_ctx)
    ctx.result = result_ctx.output
    return ctx

graph = Graph()
graph.add_node("preprocessor", preprocessor)
graph.add_node("specialist", transform_context)  # Wrapper function
graph.add_node("postprocessor", postprocessor)
```

## Benefits

1. **Simpler Mental Model**
   - One concept: composable executables
   - No need to learn multiple abstractions

2. **Smaller Codebase**
   - 46% less code to maintain
   - Easier to understand and debug

3. **Emergent Complexity**
   - Complex behaviors emerge from simple composition
   - No special cases or conditional logic

4. **Natural Subgraphs**
   - No special API for subgraphs
   - Just compose graphs like any other executable

5. **Better Testability**
   - Everything has the same interface
   - Mock/stub any level easily

6. **Reusability**
   - Build graphs once, reuse everywhere
   - Compose in different ways

## Migration Guide

### From v0.1.0 to v0.2.0

#### Breaking Changes

1. **Imports changed**
   ```python
   # Old
   from thinkagain import Conditional, Switch, Loop

   # New - use Graph with conditional edges instead
   from thinkagain import Graph
   ```

2. **Execution API changed**
   ```python
   # Old
   result = pipeline.run(ctx)        # sync
   result = await pipeline.arun(ctx) # async

   # New
   result = pipeline(ctx)            # sync
   result = await pipeline.arun(ctx) # async (preferred)
   ```

3. **Control flow changed**
   ```python
   # Old - Conditional
   pipeline = (
       worker1
       >> Conditional(
           condition=lambda ctx: ctx.score > 0.5,
           true_branch=worker2,
           false_branch=worker3
       )
   )

   # New - Graph with conditional edge
   graph = Graph()
   graph.add_node("w1", worker1)
   graph.add_node("w2", worker2)
   graph.add_node("w3", worker3)
   graph.add_edge("w1", "w2")  # Add logic node if needed
   graph.add_conditional_edge(
       "w2",
       route=lambda ctx: "high" if ctx.score > 0.5 else "low",
       paths={"high": "w2", "low": "w3"}
   )
   ```

4. **Steps API removed**
   ```python
   # Old
   for step in graph.steps(ctx):
       print(step.node, step.ctx)

   # New - use logging instead
   ctx.log_enabled = True
   result = await graph.arun(ctx)
   for entry in result.history:
       print(entry)
   ```

#### Non-Breaking Changes

- `Worker` now extends `Executable` (implementation detail)
- `Pipeline` now extends `Graph` (still works the same)
- `>>` operator still works exactly the same way

## Examples

See:
- [examples/subgraph_composition.py](examples/subgraph_composition.py) - Multi-agent system with subgraphs
- [examples/graph_examples.py](examples/graph_examples.py) - Graph with cycles (needs update)
- [examples/pipeline_examples.py](examples/pipeline_examples.py) - Sequential pipelines (needs update)

## Future Considerations

Possible additions (if needed):

1. **Parallel Execution** - Add back as a special Worker type
2. **Debugging Tools** - Better introspection and visualization
3. **Persistence** - Checkpointing and resume
4. **Stream Processing** - Yield results during execution

---