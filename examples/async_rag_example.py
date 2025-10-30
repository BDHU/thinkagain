"""
Async RAG Pipeline Example

Demonstrates how to use async pipeline components for non-blocking operations:
- AsyncWorker for async operations
- AsyncPipeline for sequential async execution
- AsyncSwitch for conditional branching
- AsyncLoop for iterative async execution

Run with: python async_rag_example.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Context
from src.core.async_pipeline import AsyncWorker, AsyncPipeline, AsyncSwitch, AsyncLoop
from src.core.parallel import Parallel


# ============================================================================
# Async Workers (simulating async I/O operations)
# ============================================================================

class AsyncVectorDBWorker(AsyncWorker):
    """Simulates async vector database search."""

    async def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching vector DB for: {ctx.query}")

        # Simulate async database query
        await asyncio.sleep(0.1)

        # Mock results
        ctx.documents = [
            f"Doc 1 about {ctx.query}",
            f"Doc 2 about {ctx.query}",
            f"Doc 3 about {ctx.query}",
        ]

        ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} documents")
        return ctx


class AsyncWebSearchWorker(AsyncWorker):
    """Simulates async web search."""

    async def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching web for: {ctx.query}")

        # Simulate async HTTP request
        await asyncio.sleep(0.2)

        # Add web results
        web_docs = [
            f"Web result 1 for {ctx.query}",
            f"Web result 2 for {ctx.query}",
        ]

        # Initialize documents if not present
        if not hasattr(ctx, 'documents') or ctx.documents is None:
            ctx.documents = []

        ctx.documents.extend(web_docs)

        ctx.log(f"[{self.name}] Added {len(web_docs)} web results")
        return ctx


class AsyncRerankerWorker(AsyncWorker):
    """Simulates async reranking."""

    async def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Reranking {len(ctx.documents)} documents")

        # Simulate async reranking model
        await asyncio.sleep(0.1)

        # Keep top N
        top_n = ctx.get("rerank_top_n", 2)
        ctx.documents = ctx.documents[:top_n]

        ctx.log(f"[{self.name}] Kept top {len(ctx.documents)} documents")
        return ctx


class AsyncGeneratorWorker(AsyncWorker):
    """Simulates async LLM generation."""

    async def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Generating answer for: {ctx.query}")

        # Simulate async LLM API call
        await asyncio.sleep(0.3)

        # Generate answer
        ctx.answer = f"Answer to '{ctx.query}' based on {len(ctx.documents)} documents"

        ctx.log(f"[{self.name}] Generated answer")
        return ctx


class AsyncQueryRefinerWorker(AsyncWorker):
    """Simulates async query refinement."""

    async def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Refining query: {ctx.query}")

        # Simulate async LLM call for query refinement
        await asyncio.sleep(0.1)

        # Refine query
        ctx.query = f"{ctx.query} (refined)"
        retry_count = ctx.get("retry_count", 0)
        ctx.retry_count = retry_count + 1

        ctx.log(f"[{self.name}] Refined query to: {ctx.query}")
        return ctx


# ============================================================================
# Example 1: Simple Async Pipeline
# ============================================================================

async def example_simple_async_pipeline():
    print("=" * 80)
    print("EXAMPLE 1: Simple Async Pipeline")
    print("=" * 80)
    print()

    # Create async workers
    vector_db = AsyncVectorDBWorker()
    reranker = AsyncRerankerWorker()
    generator = AsyncGeneratorWorker()

    # Build pipeline using >> operator (just like sync!)
    pipeline = vector_db >> reranker >> generator

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute async
    ctx = Context(query="What is machine learning?", rerank_top_n=2)
    result = await pipeline(ctx)

    print("Result:")
    print(f"  Answer: {result.answer}")
    print()

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 2: Async Pipeline with Switch
# ============================================================================

async def example_async_switch():
    print("=" * 80)
    print("EXAMPLE 2: Async Pipeline with Switch")
    print("=" * 80)
    print()

    # Create workers
    vector_db = AsyncVectorDBWorker()
    web_search = AsyncWebSearchWorker()
    reranker = AsyncRerankerWorker()
    generator = AsyncGeneratorWorker()

    # Build pipeline with async switch
    pipeline = (
        vector_db
        >> AsyncSwitch(name="quality_check")
            .case(lambda ctx: len(ctx.documents) >= 3, reranker)
            .set_default(web_search >> reranker)
        >> generator
    )

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute
    ctx = Context(query="What is AI?", rerank_top_n=3)
    result = await pipeline(ctx)

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 3: Async Pipeline with Loop
# ============================================================================

async def example_async_loop():
    print("=" * 80)
    print("EXAMPLE 3: Async Pipeline with Loop")
    print("=" * 80)
    print()

    # Create workers
    vector_db = AsyncVectorDBWorker()
    query_refiner = AsyncQueryRefinerWorker()
    generator = AsyncGeneratorWorker()

    # Build pipeline with retry loop
    pipeline = (
        vector_db
        >> AsyncLoop(
            condition=lambda ctx: len(ctx.documents) < 5 and ctx.get("retry_count", 0) < 2,
            body=query_refiner >> vector_db,
            max_iterations=3,
            name="retry_retrieval"
        )
        >> generator
    )

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute
    ctx = Context(query="quantum computing", retry_count=0)
    result = await pipeline(ctx)

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 4: Parallel Node Dependencies (using Parallel)
# ============================================================================

async def example_parallel_dependencies():
    print("=" * 80)
    print("EXAMPLE 4: Parallel Node Dependencies")
    print("=" * 80)
    print()

    # Create workers
    query_refiner = AsyncQueryRefinerWorker()
    vector_db = AsyncVectorDBWorker()
    web_search = AsyncWebSearchWorker()
    reranker = AsyncRerankerWorker()
    generator = AsyncGeneratorWorker()

    # Build pipeline with parallel dependencies
    # Both vector_db and web_search depend on query_refiner
    # They run in parallel, then results merge into reranker
    #
    #       query_refiner
    #          /      \
    #   vector_db    web_search   (parallel)
    #          \      /
    #          reranker
    #            |
    #         generator
    #
    pipeline = (
        query_refiner
        >> Parallel([vector_db, web_search])  # Both run in parallel!
        >> reranker
        >> generator
    )

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute
    ctx = Context(query="What is quantum computing?", retry_count=0, rerank_top_n=3)
    result = await pipeline(ctx)

    print(f"Result: {result.answer}")
    print()

    print("Execution History:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Example 5: Multiple Async Pipelines Concurrently
# ============================================================================

async def example_concurrent_pipelines():
    print("=" * 80)
    print("EXAMPLE 5: Run Multiple Pipelines Concurrently")
    print("=" * 80)
    print()

    # Create async workers
    vector_db = AsyncVectorDBWorker()
    reranker = AsyncRerankerWorker()
    generator = AsyncGeneratorWorker()

    # Build pipeline
    pipeline = vector_db >> reranker >> generator

    # Run multiple queries concurrently
    queries = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural networks?"
    ]

    print(f"Running {len(queries)} queries concurrently...\n")

    # Execute all pipelines concurrently
    tasks = [
        pipeline(Context(query=query, rerank_top_n=2))
        for query in queries
    ]
    results = await asyncio.gather(*tasks)

    # Print results
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        print(f"Answer: {result.answer}")
        print()


# ============================================================================
# Example 6: Complex Nested Async Pipeline
# ============================================================================

async def example_complex_async_pipeline():
    print("=" * 80)
    print("EXAMPLE 6: Complex Nested Async Pipeline")
    print("=" * 80)
    print()

    # Create workers
    vector_db = AsyncVectorDBWorker()
    web_search = AsyncWebSearchWorker()
    reranker = AsyncRerankerWorker()
    query_refiner = AsyncQueryRefinerWorker()
    generator = AsyncGeneratorWorker()

    # Build complex pipeline with nested control flow
    retrieval_with_retry = (
        vector_db
        >> AsyncLoop(
            condition=lambda ctx: len(ctx.documents) < 4,
            body=query_refiner >> vector_db,
            max_iterations=2,
            name="retry_loop"
        )
    )

    quality_check = AsyncSwitch(name="quality_check") \
        .case(lambda ctx: len(ctx.documents) >= 5, reranker) \
        .set_default(web_search >> reranker)

    pipeline = retrieval_with_retry >> quality_check >> generator

    print("Pipeline Structure:")
    print(pipeline.visualize())
    print()

    # Execute
    ctx = Context(query="What is neural networks?", retry_count=0, rerank_top_n=3)
    result = await pipeline(ctx)

    print("Final Answer:")
    print(f"  {result.answer}")
    print()

    print("Full Execution Trace:")
    for log in result.history:
        print(f"  {log}")
    print()


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all async examples."""
    await example_simple_async_pipeline()
    print("\n")

    await example_async_switch()
    print("\n")

    await example_async_loop()
    print("\n")

    await example_parallel_dependencies()
    print("\n")

    await example_concurrent_pipelines()
    print("\n")

    await example_complex_async_pipeline()


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())

    # Example: Run async pipeline synchronously (from non-async context)
    print("\n")
    print("=" * 80)
    print("BONUS: Run Async Pipeline from Sync Context (using .run())")
    print("=" * 80)
    print()

    vector_db = AsyncVectorDBWorker()
    reranker = AsyncRerankerWorker()
    generator = AsyncGeneratorWorker()

    pipeline = vector_db >> reranker >> generator
    ctx = Context(query="What is AI?", rerank_top_n=2)

    # This works from non-async context (no event loop running)
    result = pipeline.run(ctx)
    print(f"Answer: {result.answer}")
