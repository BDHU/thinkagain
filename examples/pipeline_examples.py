"""
Pipeline Examples - RAG Workflows

Demonstrates Pipeline, Switch, and Loop components with both sync and async execution.
Shows progression from basic to advanced patterns.

Run with: python examples/pipeline_examples.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Pipeline, Switch, Loop


# ============================================================================
# Workers (support both sync and async)
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

    async def acall(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching for: {ctx.query}")
        await asyncio.sleep(0.1)  # Simulate async DB query
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

    async def acall(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Reranking {len(ctx.documents)} documents")
        await asyncio.sleep(0.1)
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

    async def acall(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Generating answer")
        await asyncio.sleep(0.2)
        ctx.answer = f"Answer to '{ctx.query}' using {len(ctx.documents)} docs"
        return ctx


class WebSearch(Worker):
    """Simulates web search (fallback)."""

    def __call__(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching web")
        if not hasattr(ctx, 'documents'):
            ctx.documents = []
        ctx.documents.extend([f"Web result 1", f"Web result 2"])
        ctx.log(f"[{self.name}] Added 2 web results")
        return ctx

    async def acall(self, ctx: Context) -> Context:
        ctx.log(f"[{self.name}] Searching web")
        await asyncio.sleep(0.15)
        if not hasattr(ctx, 'documents'):
            ctx.documents = []
        ctx.documents.extend([f"Web result 1", f"Web result 2"])
        ctx.log(f"[{self.name}] Added 2 web results")
        return ctx


class QueryRefiner(Worker):
    """Refines query for better retrieval."""

    def __call__(self, ctx: Context) -> Context:
        original = ctx.query
        ctx.query = f"{original} detailed"
        ctx.log(f"[{self.name}] Refined query: {original} -> {ctx.query}")
        return ctx

    async def acall(self, ctx: Context) -> Context:
        original = ctx.query
        ctx.query = f"{original} detailed"
        ctx.log(f"[{self.name}] Refined query: {original} -> {ctx.query}")
        return ctx


# ============================================================================
# Example 1: Basic Pipeline (Sync)
# ============================================================================

def example_basic_sync():
    print("=" * 60)
    print("Example 1: Basic Pipeline (Sync)")
    print("=" * 60)

    # Build pipeline with >> operator
    pipeline = VectorDB() >> Reranker() >> Generator()

    # Execute synchronously
    ctx = Context(query="What is machine learning?", top_n=2)
    result = pipeline.run(ctx)

    print(f"\nAnswer: {result.answer}\n")


# ============================================================================
# Example 2: Basic Pipeline (Async)
# ============================================================================

async def example_basic_async():
    print("=" * 60)
    print("Example 2: Basic Pipeline (Async)")
    print("=" * 60)

    # Same pipeline, async execution
    pipeline = VectorDB() >> Reranker() >> Generator()

    # Execute asynchronously
    ctx = Context(query="What is machine learning?", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}\n")


# ============================================================================
# Example 3: Conditional Branching with Switch
# ============================================================================

async def example_switch():
    print("=" * 60)
    print("Example 3: Conditional Branching")
    print("=" * 60)

    # If we get enough docs, skip web search. Otherwise, add web results.
    pipeline = (
        VectorDB()
        >> Switch(name="quality_check")
            .case(lambda ctx: len(ctx.documents) >= 3, Reranker())
            .set_default(WebSearch() >> Reranker())
        >> Generator()
    )

    ctx = Context(query="What is AI?", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}\n")


# ============================================================================
# Example 4: Loop for Iterative Refinement
# ============================================================================

async def example_loop():
    print("=" * 60)
    print("Example 4: Loop for Iterative Refinement")
    print("=" * 60)

    # Retry retrieval with refined query until we have enough docs
    pipeline = (
        VectorDB()
        >> Loop(
            condition=lambda ctx: len(ctx.documents) < 2,
            body=QueryRefiner() >> VectorDB(),
            max_iterations=2,
            name="retry_loop"
        )
        >> Generator()
    )

    ctx = Context(query="AI", top_n=2)
    result = await pipeline.arun(ctx)

    print(f"\nAnswer: {result.answer}\n")


# ============================================================================
# Example 5: Concurrent Execution (Key Async Benefit!)
# ============================================================================

async def example_concurrent():
    print("=" * 60)
    print("Example 5: Processing Multiple Queries Concurrently")
    print("=" * 60)

    pipeline = VectorDB() >> Reranker() >> Generator()

    queries = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural networks?"
    ]

    print(f"Processing {len(queries)} queries concurrently...\n")

    # This is where async shines - all I/O operations happen concurrently!
    tasks = [pipeline.arun(Context(query=q, top_n=2)) for q in queries]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result.answer}\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    # Sync example
    example_basic_sync()
    print()

    # Async examples
    await example_basic_async()
    print()

    await example_switch()
    print()

    await example_loop()
    print()

    await example_concurrent()


if __name__ == "__main__":
    asyncio.run(main())
