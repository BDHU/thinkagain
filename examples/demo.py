"""Demo of dynamic Ray-style API with .go() calls.

This demonstrates the new dynamic execution model:
- No @jit needed - graphs built dynamically at runtime
- .go() calls return ObjectRef immediately (non-blocking)
- Direct function composition with ObjectRefs
- Automatic background optimization
- Full async/await support
"""

import asyncio
from dataclasses import dataclass

import thinkagain as ta


@dataclass
class RAGState:
    """State for RAG pipeline - plain dataclass."""

    query: str
    documents: list[str] | None = None
    answer: str = ""
    quality: float = 0.0


# ============================================================================
# Pure Functions (@op)
# ============================================================================


@ta.op
async def retrieve_docs(query: str) -> list[str]:
    """Retrieve documents for a query."""
    await asyncio.sleep(0.01)  # Simulate async I/O
    return [f"Document {i} about '{query}'" for i in range(1, 4)]


@ta.op
async def combine_docs(docs: list[str]) -> str:
    """Combine documents into a context string."""
    return "\\n\\n".join(docs)


@ta.op
async def generate_answer(context: str, query: str) -> str:
    """Generate answer from context."""
    await asyncio.sleep(0.02)
    return f"Answer to '{query}' based on: {context[:50]}..."


@ta.op
async def evaluate_quality(answer: str) -> float:
    """Evaluate answer quality."""
    await asyncio.sleep(0.01)
    return 0.85 if len(answer) > 20 else 0.5


@ta.op
async def build_state(
    query: str, docs: list[str], answer: str, quality: float
) -> RAGState:
    """Build final RAG state."""
    return RAGState(query=query, documents=docs, answer=answer, quality=quality)


# ============================================================================
# Pipelines (No @jit needed!)
# ============================================================================


async def rag_pipeline_sequential(query: str) -> RAGState:
    """Sequential RAG pipeline - simple and clear.

    Each step waits for the previous one to complete.
    """
    # Retrieve documents
    docs = await retrieve_docs.go(query)

    # Combine into context
    context = await combine_docs.go(docs)

    # Generate answer
    answer = await generate_answer.go(context, query)

    # Evaluate quality
    quality = await evaluate_quality.go(answer)

    # Build final state
    return await build_state.go(query, docs, answer, quality)


async def rag_pipeline_parallel(query: str) -> RAGState:
    """Parallel RAG pipeline - maximum performance.

    Submits all tasks immediately (non-blocking) and lets the scheduler
    handle dependencies. The scheduler automatically runs independent
    tasks concurrently.
    """
    # Submit all tasks (non-blocking)
    docs_ref = retrieve_docs.go(query)
    context_ref = combine_docs.go(docs_ref)  # Depends on docs_ref
    answer_ref = generate_answer.go(context_ref, query)  # Depends on context_ref
    quality_ref = evaluate_quality.go(answer_ref)  # Depends on answer_ref

    # Wait for all results
    docs = await docs_ref
    answer = await answer_ref
    quality = await quality_ref

    # Build final state
    return await build_state.go(query, docs, answer, quality)


async def multi_query_pipeline(queries: list[str]) -> list[RAGState]:
    """Process multiple queries in parallel.

    Demonstrates natural parallelism with .go() - submit all queries
    immediately and wait for results.
    """
    # Submit all queries (fully parallel)
    refs = [rag_pipeline_parallel(query) for query in queries]

    # Wait for all results
    return await asyncio.gather(*refs)


# ============================================================================
# Stateful Services (@service)
# ============================================================================


@ta.service()
class QueryCache:
    """Simple cache demonstrating mutable actor state."""

    def __init__(self):
        self.cache: dict[str, RAGState] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, query: str) -> RAGState | None:
        """Get cached result."""
        if query in self.cache:
            self.hits += 1
            return self.cache[query]
        self.misses += 1
        return None

    async def put(self, query: str, result: RAGState):
        """Cache a result."""
        self.cache[query] = result

    async def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
        }


async def cached_rag_pipeline(cache, query: str) -> RAGState:
    """RAG pipeline with caching.

    Demonstrates mutable actor state - cache persists across calls.
    """
    # Check cache first
    cached_ref = cache.get.go(query)
    cached = await cached_ref

    if cached is not None:
        print(f"  Cache HIT for: {query}")
        return cached

    print(f"  Cache MISS for: {query}")

    # Compute result
    result = await rag_pipeline_parallel(query)

    # Store in cache (mutable state update)
    await cache.put.go(query, result)

    return result


# ============================================================================
# Main Demo
# ============================================================================


async def main():
    """Run demos."""
    print("=" * 70)
    print("DYNAMIC RAY-STYLE API DEMO")
    print("=" * 70)

    # Create mesh
    mesh = ta.Mesh([ta.CpuDevice(0)])

    with mesh:
        print("\\n" + "=" * 70)
        print("Demo 1: Sequential Pipeline")
        print("=" * 70)
        result = await rag_pipeline_sequential("What is Python?")
        print(f"Query: {result.query}")
        print(f"Documents: {len(result.documents or [])} docs")
        print(f"Answer: {result.answer}")
        print(f"Quality: {result.quality}")

        print("\\n" + "=" * 70)
        print("Demo 2: Parallel Pipeline (with .go())")
        print("=" * 70)
        result = await rag_pipeline_parallel("What is machine learning?")
        print(f"Query: {result.query}")
        print(f"Answer: {result.answer}")
        print(f"Quality: {result.quality}")

        print("\\n" + "=" * 70)
        print("Demo 3: Multiple Queries in Parallel")
        print("=" * 70)
        queries = [
            "What is distributed computing?",
            "What is quantum computing?",
            "What is cloud computing?",
        ]
        results = await multi_query_pipeline(queries)
        for r in results:
            print(f"  - {r.query}: quality={r.quality}")

        print("\\n" + "=" * 70)
        print("Demo 4: Mutable Actor State (Caching)")
        print("=" * 70)

        # Create cache actor
        cache = QueryCache.init()  # type: ignore[attr-defined]

        # First call - cache miss
        r1 = await cached_rag_pipeline(cache, "What is Python?")
        print(f"Result 1: {r1.answer}")

        # Second call - cache hit!
        r2 = await cached_rag_pipeline(cache, "What is Python?")
        print(f"Result 2: {r2.answer}")

        # Different query - cache miss
        r3 = await cached_rag_pipeline(cache, "What is Java?")
        print(f"Result 3: {r3.answer}")

        # Check stats (mutable state)
        stats_ref = cache.stats.go()
        stats = await stats_ref
        print(f"\\nCache stats: {stats}")

    print("\\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    print("\\nKey Takeaways:")
    print("  - No @jit needed - dynamic graph building")
    print("  - .go() returns ObjectRef immediately (non-blocking)")
    print("  - Natural parallelism by submitting multiple .go() calls")
    print("  - Mutable actor state with @service")
    print("  - Automatic background optimization")


if __name__ == "__main__":
    asyncio.run(main())
