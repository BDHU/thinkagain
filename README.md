# Minimal Agent Framework

A simple, debuggable framework for building agent pipelines with explicit control.

## Design Principles

1. **Explicit over Implicit** - Every step is visible and controllable
2. **Full Transparency** - Inspect intermediate results at any point
3. **Easy to Debug** - Built-in history tracking and logging
4. **Minimal Abstractions** - Only what you need, nothing more

## Core Components

### Context

A container that holds state and tracks execution history as it passes through workers.

```python
from src.core import Context

# Create context with initial data
ctx = Context(query="What is ML?", top_k=5)

# Workers modify the context
ctx.documents = ["doc1", "doc2"]
ctx.log("Retrieved documents")

# Inspect anytime
print(ctx.documents)  # Access data
print(ctx.history)    # See execution log
print(ctx.data)       # Get all data as dict
```

### Worker

Base class for any processing unit (vector DB, LLM, reranker, etc.).

```python
from src.core import Worker, Context

class VectorDBWorker(Worker):
    def __call__(self, ctx: Context) -> Context:
        # Perform work
        results = self.search(ctx.query)

        # Update context
        ctx.documents = results
        ctx.log(f"[{self.name}] Retrieved {len(results)} docs")

        # Return modified context
        return ctx

    def search(self, query: str):
        # Your implementation
        pass
```

## Usage Examples

### Basic RAG Pipeline

```python
from src.core import Context

# Initialize workers
vector_db = VectorDBWorker()
reranker = RerankerWorker()
generator = GeneratorWorker()

# Create context
ctx = Context(query="What is machine learning?")

# Run pipeline - fully explicit
ctx = vector_db(ctx)
print(f"Retrieved: {ctx.documents}")  # Inspect

ctx = reranker(ctx)
print(f"Reranked: {ctx.documents}")   # Inspect

ctx = generator(ctx)
print(f"Answer: {ctx.answer}")        # Inspect
print(f"History: {ctx.history}")      # See full execution log
```

### Functional Composition

```python
# Chain workers for cleaner code
ctx = Context(query="What is ML?")
ctx = generator(reranker(vector_db(ctx)))

# Still fully inspectable!
print(ctx.answer)
print(ctx.history)
```

### Step-by-Step Debugging

```python
ctx = Context(query="What is ML?")

# Run one step at a time
ctx = vector_db(ctx)
# Pause and inspect
if len(ctx.documents) < 3:
    ctx.top_k = 10  # Adjust parameters
    ctx = vector_db(ctx)  # Re-run

# Continue when satisfied
ctx = reranker(ctx)
ctx = generator(ctx)
```

## Running the Example

```bash
python examples/simple_rag.py
```

This demonstrates three different usage patterns:
1. Full pipeline with verbose output
2. Step-by-step execution with inspection
3. Functional composition style

## Why This Design?

### Problem with Existing Frameworks

Many agent frameworks become "black boxes":
- Hard to debug when things go wrong
- Difficult to inspect intermediate results
- Too much magic/abstraction
- Complex tool calling mechanisms

### Our Solution

**Context-passing pattern**:
- Every step is explicit
- Full visibility into what's happening
- Easy to modify and extend
- Simple to understand and debug

**Workers as first-class entities**:
- Vector DBs, LLMs, rerankers are all workers
- Uniform interface (`__call__`)
- Composable and reusable
- Easy to test individually

## Computation Graph / Pipeline DSL

**NEW**: Build pipelines with explicit control flow using a functional DSL!

### Simple Pipeline Composition

```python
from src.core import Pipeline, Switch, Loop

# Compose workers using >> operator
pipeline = vector_db >> reranker >> generator

# Graph structure is captured BEFORE execution
print(pipeline.to_dict())  # Full graph definition
print(pipeline.visualize())  # ASCII visualization

# Execute
ctx = pipeline.run(Context(query="What is ML?"))
```

### Conditional Branching (if/else)

```python
from src.core import Conditional

pipeline = (
    retrieve
    >> Conditional(
        condition=lambda ctx: len(ctx.documents) >= 2,
        true_branch=rerank,
        false_branch=fallback >> rerank
    )
    >> generate
)
```

### Multi-way Branching (if/elif/else)

```python
from src.core import Switch

pipeline = (
    retrieve
    >> Switch(name="quality_check")
        .case(lambda ctx: len(ctx.documents) >= 5, high_quality_path)
        .case(lambda ctx: len(ctx.documents) >= 2, medium_quality_path)
        .set_default(fallback_path)
    >> generate
)
```

### Loops

```python
from src.core import Loop

pipeline = (
    retrieve
    >> Loop(
        condition=lambda ctx: len(ctx.documents) < 2,
        body=refine_query >> retrieve,
        max_iterations=3
    )
    >> generate
)
```

### Complex Nested Pipelines

```python
# Sub-pipelines can be composed
retrieval_with_retry = retrieve >> Loop(...)
quality_check = Switch(...).case(...).set_default(...)

# Compose them together
pipeline = retrieval_with_retry >> quality_check >> generate

# Full graph structure is captured!
print(pipeline.visualize())
```

### Graph Visualization Example

```
Pipeline: rag_pipeline
├── vector_db
├── Switch: quality_check
│   ├── CASE 1: reranker
│   ├── CASE 2: reranker
│   └── DEFAULT: pipeline
│       ├── web_search
│       └── reranker
└── generator
```

## Next Steps

This minimal implementation focuses on synchronous execution. Future enhancements:

- [ ] Async/await support for concurrent workers
- [ ] Parallel execution branches
- [ ] Event loop for message-based coordination
- [ ] State management and memory
- [ ] Built-in observability (tracing, metrics)
- [ ] Tool registry and discovery
- [ ] GraphViz/Mermaid export for visualization

## Philosophy

> "Simple is better than complex. Explicit is better than implicit."
>
> — The Zen of Python

Start simple, add complexity only when needed.
