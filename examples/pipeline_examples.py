"""
Pipeline Examples - RAG Workflows

Demonstrates simple sequential pipelines using the >> operator.
Shows that Pipeline is just syntactic sugar for Graph.

This example shows:
- Sequential composition with >> operator
- Pipeline class for explicit sequential graphs
- Both sync and async execution
- Concurrent execution of multiple pipelines

Run with: python examples/pipeline_examples.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Pipeline


# ============================================================================
# Workers (sync with auto-wrapped async)
# ============================================================================

class VectorDB(Worker):
    """Simulates vector database search."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching for: {ctx.query}")
        ctx.documents = [
            f"Doc 1 about {ctx.query}",
            f"Doc 2 about {ctx.query}",
            f"Doc 3 about {ctx.query}",
        ]
        ctx.log(f"[{self.name}] Found {len(ctx.documents)} documents")
        return ctx


class Reranker(Worker):
    """Simulates reranking."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Reranking {len(ctx.documents)} documents")
        top_n = ctx.get("top_n", 2)
        ctx.documents = ctx.documents[:top_n]
        ctx.log(f"[{self.name}] Kept top {len(ctx.documents)}")
        return ctx


class Generator(Worker):
    """Simulates LLM generation."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Generating answer")
        ctx.answer = f"Answer to '{ctx.query}' using {len(ctx.documents)} docs"
        return ctx


class QueryRefiner(Worker):
    """Refines query for better retrieval."""

    def __call__(self, ctx: Context) -> Context:
        original = ctx.query
        ctx.query = f"{original} (detailed)"
        ctx.log(f"[{self.name}] Refined query: {original} -> {ctx.query}")
        return ctx


# ============================================================================
# Example 1: Basic Pipeline with >> Operator
# ============================================================================

async def example_basic_pipeline():
    print("=" * 70)
    print("Example 1: Basic Pipeline (>> Operator)")
    print("=" * 70)

    # Build pipeline with >> operator - most concise!
    pipeline = VectorDB() >> Reranker() >> Generator()

    print(f"\nPipeline type: {type(pipeline).__name__}")
    print(f"Pipeline is a Graph: {pipeline.__class__.__bases__}")
    print(f"Nodes: {len(pipeline.nodes)}")

    # Execute
    ctx = Context(query="What is machine learning?", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}")
    print(f"Execution path: {' → '.join(result.execution_path)}")
    print()


# ============================================================================
# Example 2: Explicit Pipeline Construction
# ============================================================================

async def example_explicit_pipeline():
    print("=" * 70)
    print("Example 2: Explicit Pipeline Construction")
    print("=" * 70)

    # Alternative: explicit Pipeline construction
    pipeline = Pipeline([
        VectorDB(),
        Reranker(),
        Generator()
    ], name="rag_pipeline")

    print(f"\nPipeline name: {pipeline.name}")
    print(f"Nodes: {list(pipeline.nodes.keys())}")

    # Visualize
    print("\nMermaid diagram:")
    print(pipeline.visualize())

    # Execute
    ctx = Context(query="What is deep learning?", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}")
    print()


# ============================================================================
# Example 3: Async is Primary (Sync shown in separate script)
# ============================================================================

async def example_async_primary():
    print("=" * 70)
    print("Example 3: Async-First Execution")
    print("=" * 70)

    # Same pipeline
    pipeline = VectorDB() >> Reranker() >> Generator()

    print("\nAsync is the primary execution method:")
    print("  await pipeline.arun(ctx)")
    print("\nFor sync usage, call from non-async context:")
    print("  result = pipeline(ctx)  # wraps in asyncio.run()")

    # Execute async
    ctx = Context(query="What is AI?", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}")
    print()


# ============================================================================
# Example 4: Concurrent Execution (Key Async Benefit!)
# ============================================================================

async def example_concurrent_execution():
    print("=" * 70)
    print("Example 4: Processing Multiple Queries Concurrently")
    print("=" * 70)

    pipeline = VectorDB() >> Reranker() >> Generator()

    queries = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural networks?"
    ]

    print(f"\nProcessing {len(queries)} queries concurrently...\n")

    # This is where async shines - all queries run concurrently!
    tasks = [pipeline.arun(Context(query=q, top_n=2)) for q in queries]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result.answer}\n")


# ============================================================================
# Example 5: Pipeline Composition
# ============================================================================

async def example_pipeline_composition():
    print("=" * 70)
    print("Example 5: Composing Pipelines")
    print("=" * 70)

    # Create reusable pipeline segments
    retrieval_pipeline = QueryRefiner() >> VectorDB()
    generation_pipeline = Reranker() >> Generator()

    # Compose them
    full_pipeline = retrieval_pipeline >> generation_pipeline

    print(f"\nComposed pipeline has {len(full_pipeline.nodes)} nodes")
    print("Pipelines compose naturally with >> operator!")

    # Execute
    ctx = Context(query="AI", top_n=2)
    result = await full_pipeline.arun(ctx)

    print(f"\nOriginal query: AI")
    print(f"Refined query: {result.query}")
    print(f"Answer: {result.answer}")
    print()


# ============================================================================
# Main
# ============================================================================

async def main():
    await example_basic_pipeline()
    await example_explicit_pipeline()
    await example_async_primary()
    await example_concurrent_execution()
    await example_pipeline_composition()

    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("1. Pipeline is just a Graph with sequential auto-wiring")
    print("2. >> operator is the most concise way to build pipelines")
    print("3. Everything is async-first, sync wraps it")
    print("4. Pipelines compose naturally with >>")
    print("5. Use async for concurrent execution of multiple pipelines")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
