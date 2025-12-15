# ThinkAgain Design (Historical Notes)

## Overview

A minimal, debuggable agent framework focused on explicit control and transparency.
The framework originally captured computation graphs **before execution** via an
explicit `Graph`/`Pipeline` API. The current core library is intentionally much
smaller and focuses on:

- `Context` – state and history
- `Executable` – async components
- `Node` – declarative wrapper
- `LazyContext` – lazy chaining and materialization
- `run` – execution helper

The sections below describe the earlier, graph-centric design and are kept for
historical context and future experimentation.

## Core Architecture

### 1. Context (State Container)

```python
class Context:
    - _data: dict           # Holds all pipeline state
    - _history: list[str]   # Execution log for debugging
```

**Purpose**: Passes state between workers while tracking execution history.

**Key Features**:
- Dictionary-like attribute access
- Built-in logging via `ctx.log()`
- Full history tracking
- No hidden state

### 2. Worker (Computation Unit)

```python
class Worker:
    - name: str
    - __call__(ctx: Context) -> Context
    - __rshift__(other) -> Pipeline  # >> operator
    - to_dict() -> dict              # Export structure
```

**Purpose**: Base class for any processing unit (vector DB, LLM, reranker, etc.)

**Key Features**:
- Uniform interface: `ctx = worker(ctx)`
- Composable via `>>`operator
- Self-describing via `to_dict()`
- Can represent services, algorithms, or LLMs

### 3. Pipeline (Composition)

```python
class Pipeline:
    - name: str
    - nodes: list  # Workers, Switches, Loops
    - __call__(ctx: Context) -> Context
    - visualize() -> str
    - to_dict() -> dict
```

**Purpose**: Sequential composition of workers and control flow.

**Key Features**:
- Created via `worker1 >> worker2 >> worker3`
- Captures graph structure before execution
- Can be nested (pipelines in pipelines)
- Exports to dict/JSON for inspection

### 4. Switch (Multi-way Conditional)

```python
class Switch:
    - name: str
    - cases: list[tuple[condition, branch]]
    - default: branch
    - case(condition, branch) -> Switch  # Builder
    - set_default(branch) -> Switch      # Builder
```

**Purpose**: Multi-way branching (if/elif/else).

**Key Features**:
- Evaluates conditions in order
- Executes first matching branch
- Falls back to default if no match
- Builder pattern for readability

**Usage**:
```python
Switch(name="routing")
    .case(lambda ctx: condition1, branch1)
    .case(lambda ctx: condition2, branch2)
    .set_default(default_branch)
```

### 5. Conditional (Binary Switch)

```python
class Conditional(Switch):
    - condition: Callable
    - true_branch: Worker/Pipeline
    - false_branch: Worker/Pipeline
```

**Purpose**: Simple if/else branching (convenience wrapper).

**Usage**:
```python
Conditional(
    condition=lambda ctx: len(ctx.documents) >= 2,
    true_branch=reranker,
    false_branch=fallback
)
```

### 6. Loop (Iteration)

```python
class Loop:
    - name: str
    - condition: Callable
    - body: Worker/Pipeline
    - max_iterations: int
```

**Purpose**: Repeat execution while condition is true.

**Key Features**:
- Safety limit via `max_iterations`
- Logs each iteration
- Can nest loops and conditionals

**Usage**:
```python
Loop(
    condition=lambda ctx: len(ctx.documents) < 2,
    body=refine_query >> retrieve,
    max_iterations=3
)
```

## Computation Graph Capture

### Static Definition (Before Execution)

The graph structure is captured **declaratively** using the `>>` operator:

```python
pipeline = (
    retrieve
    >> Switch(name="quality")
        .case(lambda ctx: len(ctx.documents) >= 5, high_path)
        .case(lambda ctx: len(ctx.documents) >= 2, med_path)
        .set_default(low_path)
    >> generate
)

# Graph is fully defined - nothing has executed yet!
graph_dict = pipeline.to_dict()  # Complete graph structure
graph_viz = pipeline.visualize()  # ASCII visualization
```

### Graph Representation

Graphs are represented as nested dictionaries:

```json
{
  "type": "Pipeline",
  "name": "rag_pipeline",
  "nodes": [
    {
      "type": "Worker",
      "name": "vector_db",
      "class": "VectorDBWorker"
    },
    {
      "type": "Switch",
      "name": "quality_check",
      "cases": [
        {
          "condition": "lambda",
          "branch": {"type": "Worker", "name": "reranker"}
        }
      ],
      "default": {
        "type": "Pipeline",
        "nodes": [...]
      }
    },
    {
      "type": "Worker",
      "name": "generator",
      "class": "GeneratorWorker"
    }
  ]
}
```

### Execution Trace (Runtime)

During execution, Context tracks what actually happened:

```python
ctx = pipeline.run(Context(query="What is ML?"))

# Execution history
print(ctx.history)
# [
#   "[Pipeline] Starting: rag_pipeline",
#   "[vector_db] Retrieved 3 documents",
#   "[Switch] Evaluating: quality_check",
#   "[Switch] Case 1 evaluated to: True",
#   "[Switch] Executing case 1",
#   "[reranker] Reranked to top 2",
#   "[generator] Generated answer",
#   "[Pipeline] Completed: rag_pipeline"
# ]
```

## Control Flow Examples

### Simple Linear Pipeline

```python
pipeline = worker1 >> worker2 >> worker3
```

**Graph**:
```
Pipeline
├── worker1
├── worker2
└── worker3
```

### Conditional Pipeline

```python
pipeline = (
    worker1
    >> Conditional(
        condition=lambda ctx: ctx.value > 10,
        true_branch=worker2,
        false_branch=worker3
    )
    >> worker4
)
```

**Graph**:
```
Pipeline
├── worker1
├── Conditional
│   ├── TRUE: worker2
│   └── FALSE: worker3
└── worker4
```

### Multi-way Branching

```python
pipeline = (
    worker1
    >> Switch(name="routing")
        .case(lambda ctx: ctx.score >= 0.9, high_quality)
        .case(lambda ctx: ctx.score >= 0.5, medium_quality)
        .set_default(low_quality)
    >> worker2
)
```

**Graph**:
```
Pipeline
├── worker1
├── Switch: routing
│   ├── CASE 1: high_quality
│   ├── CASE 2: medium_quality
│   └── DEFAULT: low_quality
└── worker2
```

### Loop with Retry

```python
pipeline = (
    worker1
    >> Loop(
        condition=lambda ctx: not ctx.success and ctx.retries < 3,
        body=retry_worker,
        max_iterations=5
    )
    >> worker2
)
```

**Graph**:
```
Pipeline
├── worker1
├── Loop: retry (max=5)
│   └── Body: retry_worker
└── worker2
```

### Complex Nested Pipeline

```python
retrieval_stage = (
    vector_db
    >> Loop(
        condition=lambda ctx: len(ctx.documents) < 2,
        body=refine_query >> vector_db,
        max_iterations=3
    )
)

quality_stage = (
    Switch(name="quality")
        .case(lambda ctx: len(ctx.documents) >= 3, reranker)
        .set_default(web_search >> reranker)
)

pipeline = retrieval_stage >> quality_stage >> generator
```

**Graph**:
```
Pipeline
├── vector_db
├── Loop: retry (max=3)
│   └── Body: (sub-pipeline)
│       ├── refine_query
│       └── vector_db
├── Switch: quality
│   ├── CASE 1: reranker
│   └── DEFAULT: (sub-pipeline)
│       ├── web_search
│       └── reranker
└── generator
```

## Extension Points

### Future Enhancements
1. Add declerative graph construction.
2. Consolidate `arun` and `acall`.
3. **Caching** - Add @cached decorator for workers
4. **Graph Optimization** - Analyze and optimize before execution
5. **Visualization** - Export to GraphViz, Mermaid, etc.
6. **Distributed Execution** - Run workers on different machines
7. **Type Checking** - Validate data flow between workers
