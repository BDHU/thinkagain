"""Demo of JAX-style API with @jit boundaries and in-graph control flow.

This demonstrates the new design where:
- No Context wrapper needed
- Functions operate on plain data types (dataclasses, dicts, lists, primitives, etc.)
- @node functions can take ANY Python data as arguments
- @jit creates graph boundaries (like jax.jit)
- Control flow uses cond(), while_loop(), scan()
- Python control flow works outside @jit
- Full async/await support throughout
"""

import asyncio
from dataclasses import dataclass, replace

# New JAX-style imports
import thinkagain


@dataclass
class RAGState:
    """State for RAG pipeline - plain dataclass, no Context wrapper!"""

    query: str
    documents: list[str] = None
    answer: str = ""
    quality: float = 0.0
    retrieval_attempt: int = 0
    refinements: int = 0

    def __post_init__(self):
        if self.documents is None:
            self.documents = []


# ============================================================================
# Nodes - operate on ANY Python data types
# ============================================================================


@thinkagain.node
async def retrieve_docs(state: RAGState) -> RAGState:
    """Retrieve documents - takes a dataclass (async demo)."""
    await asyncio.sleep(0.001)  # Simulate async I/O
    attempt = state.retrieval_attempt + 1
    docs = [f"Document {i} for '{state.query}'" for i in range(1, min(5, 1 + attempt))]
    return replace(state, documents=docs, retrieval_attempt=attempt)


@thinkagain.node
async def process_dict(data: dict) -> dict:
    """Process a plain dict - no special wrapper needed!"""
    await asyncio.sleep(0.001)
    return {
        "query": data["query"],
        "result": f"Processed: {data['query']}",
        "count": data.get("count", 0) + 1,
    }


@thinkagain.node
async def add_numbers(x: int, y: int) -> int:
    """Take primitive types directly."""
    await asyncio.sleep(0.001)
    return x + y


@thinkagain.node
async def process_list(items: list[str]) -> list[str]:
    """Take lists directly."""
    await asyncio.sleep(0.001)
    return [item.upper() for item in items]


@thinkagain.node
async def rerank_docs(state: RAGState) -> RAGState:
    """Keep only top 2 documents."""
    await asyncio.sleep(0.001)
    return replace(state, documents=state.documents[:2])


@thinkagain.node
async def generate_answer(state: RAGState) -> RAGState:
    """Generate answer from documents."""
    await asyncio.sleep(0.001)  # Simulate LLM call
    if state.documents:
        docs_str = ", ".join(state.documents[:2])
        answer = f"Based on {docs_str}, the answer is..."
    else:
        answer = "No documents found."
    return replace(state, answer=answer)


@thinkagain.node
async def critique_answer(state: RAGState) -> RAGState:
    """Evaluate answer quality."""
    await asyncio.sleep(0.001)
    if len(state.documents) >= 3:
        quality = 0.9
    elif len(state.documents) == 2:
        quality = 0.7
    else:
        quality = 0.4
    return replace(state, quality=quality)


@thinkagain.node
async def refine_query(state: RAGState) -> RAGState:
    """Refine the query for better retrieval."""
    await asyncio.sleep(0.001)
    r = state.refinements + 1
    refined = f"{state.query} (refined {r}x)"
    return replace(state, query=refined, refinements=r)


# ============================================================================
# Example 1: No JIT - Regular Python (No graph building)
# ============================================================================


async def non_jit_pipeline(state: RAGState) -> RAGState:
    """Regular Python - no graph, just executes directly (async).

    Outside @jit, nodes just execute like normal async functions.
    No Context wrapper, no graph building.
    """
    print("\n=== UNCOMPILED PIPELINE ===")
    print("No graph building, just normal Python execution")

    state = await retrieve_docs(state)
    print(f"Retrieved {len(state.documents)} documents")

    # Regular Python if - works fine outside @jit
    if len(state.documents) > 2:
        print("Many docs, reranking...")
        state = await rerank_docs(state)

    state = await generate_answer(state)
    return state


# ============================================================================
# Example 2: Arbitrary Python data types
# ============================================================================


async def test_various_types():
    """Demo that nodes work with any Python types (async)."""
    print("\n=== VARIOUS DATA TYPES ===")

    # Primitives
    result = await add_numbers(5, 10)
    print(f"5 + 10 = {result}")

    # Dicts
    data = {"query": "test", "count": 0}
    result = await process_dict(data)
    print(f"Dict result: {result}")

    # Lists
    items = ["hello", "world"]
    result = await process_list(items)
    print(f"List result: {result}")

    # Dataclasses
    state = RAGState(query="test")
    result = await retrieve_docs(state)
    print(f"Dataclass result: {len(result.documents)} docs")


# ============================================================================
# Example 3: JIT with in-graph conditional
# ============================================================================


@thinkagain.jit
async def jit_pipeline(state: RAGState) -> RAGState:
    """JIT pipeline - graph is built once, then reused (async).

    Inside @jit:
    - Must use cond() instead of Python if
    - Graph is traced once
    - Subsequent calls reuse the compiled graph
    - All async operations handled automatically
    """
    state = await retrieve_docs(state)

    # Must use cond() inside @jit, not Python if
    state = await thinkagain.cond(
        lambda s: len(s.documents) > 2,  # Predicate
        rerank_docs,  # True branch
        lambda s: s,  # False branch (identity)
        state,  # Operand
    )

    state = await generate_answer(state)
    return state


# ============================================================================
# Example 4: Self-correcting RAG with while_loop
# ============================================================================


@thinkagain.jit
async def self_correcting_rag_jit(state: RAGState) -> RAGState:
    """Self-correcting RAG using in-graph while_loop (async).

    The while_loop is part of the graph - conditions are evaluated
    at runtime but the graph structure is fixed.
    """
    state = await retrieve_docs(state)
    state = await generate_answer(state)
    state = await critique_answer(state)

    # Data-dependent loop - compiled into graph!
    def should_continue(s: RAGState) -> bool:
        return s.quality < 0.8 and s.retrieval_attempt < 3

    async def retry_body(s: RAGState) -> RAGState:
        s = await refine_query(s)
        s = await retrieve_docs(s)
        s = await generate_answer(s)
        s = await critique_answer(s)
        return s

    state = await thinkagain.while_loop(should_continue, retry_body, state)
    return state


# ============================================================================
# Example 5: Hybrid - Python control flow + JIT hot paths
# ============================================================================


async def hybrid_pipeline(state: RAGState) -> RAGState:
    """Mix Python control flow (outer) with JIT regions (inner).

    This is the recommended pattern:
    - Use Python control flow for high-level orchestration
    - Use @jit for performance-critical hot paths
    """
    print("\n=== HYBRID PIPELINE ===")
    print("Python control flow + JIT hot paths")

    # Python control flow - check cache
    if state.query.startswith("cached:"):
        print("Cache hit!")
        return replace(state, answer="Cached result")

    # Compiled hot path for main processing
    state = await jit_pipeline(state)

    # More Python control flow - fallback
    if state.quality < 0.5:
        print("Quality too low, using fallback")
        state = replace(state, answer="Fallback answer")

    return state


# ============================================================================
# Example 6: Batch processing with scan
# ============================================================================


@thinkagain.jit
async def batch_process(queries: list[str]) -> list[str]:
    """Process multiple queries using scan (fixed iteration count, async).

    scan is the most efficient control flow operator because the
    iteration count is known at compile time.
    """

    def process_one(carry: int, query: str) -> tuple[int, str]:
        count = carry + 1
        result = f"Processed: {query} (#{count})"
        return count, result

    final_count, results = await thinkagain.scan(process_one, init=0, xs=queries)
    return results


# ============================================================================
# Example 7: Multiple arguments with different types
# ============================================================================


@thinkagain.node
async def merge_data(state: RAGState, extra_docs: list[str], boost: float) -> RAGState:
    """Node can take multiple arguments of different types."""
    await asyncio.sleep(0.001)
    all_docs = state.documents + extra_docs
    quality = state.quality * boost
    return replace(state, documents=all_docs, quality=quality)


@thinkagain.jit
async def multi_arg_pipeline(state: RAGState) -> RAGState:
    """Pipeline with multi-argument nodes (async)."""
    state = await retrieve_docs(state)

    # Call node with multiple different types
    extra = ["Bonus doc 1", "Bonus doc 2"]
    state = await merge_data(state, extra, 1.2)

    state = await generate_answer(state)
    return state


# ============================================================================
# Main Demo
# ============================================================================


async def main():
    print("=" * 70)
    print("JAX-STYLE API DEMO (with async/await)")
    print("Key feature: @node works with ANY Python data types!")
    print("=" * 70)

    # Test various data types
    await test_various_types()

    # Example 1: No JIT
    print("\n" + "=" * 70)
    state = RAGState(query="What is machine learning?")
    result = await non_jit_pipeline(state)
    print(f"Result: {result.answer}")
    print(f"Documents: {len(result.documents)}")

    # Example 2: JIT
    print("\n" + "=" * 70)
    print("=== JIT PIPELINE ===")
    print("Graph built once, executed efficiently")
    state = RAGState(query="What is deep learning?")
    result = await jit_pipeline(state)
    print(f"Result: {result.answer}")
    print(f"Documents: {len(result.documents)}")

    # Run again - should reuse compiled graph
    print("\nRunning JIT pipeline again (reuses graph)...")
    state2 = RAGState(query="Explain neural networks")
    result2 = await jit_pipeline(state2)
    print(f"Result: {result2.answer}")

    # Example 3: Self-correcting with while_loop
    print("\n" + "=" * 70)
    print("=== SELF-CORRECTING RAG (with while_loop) ===")
    state = RAGState(query="Complex question")
    result = await self_correcting_rag_jit(state)
    print(f"Result: {result.answer}")
    print(f"Quality: {result.quality}")
    print(f"Attempts: {result.retrieval_attempt}")
    print(f"Refinements: {result.refinements}")

    # Example 4: Hybrid
    state = RAGState(query="Test query")
    result = await hybrid_pipeline(state)
    print(f"Result: {result.answer}")

    # Test cache path
    state = RAGState(query="cached:something")
    result = await hybrid_pipeline(state)
    print(f"Cached result: {result.answer}")

    # Example 5: Batch processing with scan
    print("\n" + "=" * 70)
    print("=== BATCH PROCESSING (with scan) ===")
    queries = ["Query 1", "Query 2", "Query 3"]
    results = await batch_process(queries)
    for r in results:
        print(f"  {r}")

    # Example 6: Multiple arguments
    print("\n" + "=" * 70)
    print("=== MULTI-ARGUMENT NODES ===")
    state = RAGState(query="Multi-arg test")
    result = await multi_arg_pipeline(state)
    print(f"Result: {result.answer}")
    print(f"Total docs: {len(result.documents)}")

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("Key takeaway: Full async/await support with no Context wrapper!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
