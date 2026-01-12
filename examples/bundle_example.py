"""Example: Using Bundle for clean pipeline input management.

This example demonstrates the Bundle API for managing pipeline inputs:
- Bundle: Lightweight traced container for grouping related values
- ta.*: Runtime Bundle operations that create graph nodes
- make_inputs: Type-safe input factory
"""

import asyncio
from typing import TYPE_CHECKING

import thinkagain as ta

if TYPE_CHECKING:
    from dataclasses import dataclass

    @dataclass
    class RagInputs:
        """Type for RagInputs bundle."""

        query: str
        llm: str
        db: str
        temperature: float


# === Example 1: Basic Bundle Usage ===


@ta.node
async def retrieve_docs(query: str, db_name: str) -> list[str]:
    """Simulate document retrieval."""
    return [f"doc1 about {query}", f"doc2 about {query}"]


@ta.node
async def generate_answer(query: str, docs: list[str], llm: str) -> str:
    """Simulate answer generation."""
    docs_str = ", ".join(docs)
    return f"[{llm}] Answer for '{query}' using: {docs_str}"


@ta.jit
async def basic_rag_pipeline(inputs: ta.Bundle) -> str:
    """Simple RAG pipeline using Bundle operations."""
    # Unpack fields directly - clean and Pythonic!
    query, db, llm = await ta.unpack(inputs, "query", "db", "llm")

    # Retrieve documents
    docs = await retrieve_docs(query, db)

    # Generate answer
    answer = await generate_answer(query, docs, llm)

    return answer


# === Example 2: Complex Pipeline with Bundle Transformations ===


@ta.node
async def enrich_query(query: str, user_context: dict) -> str:
    """Enrich query with user context."""
    return f"{query} [user: {user_context.get('name', 'unknown')}]"


@ta.node
async def rerank_docs(docs: list[str], model: str) -> list[str]:
    """Rerank documents."""
    return [f"reranked({model}): {doc}" for doc in docs]


@ta.jit
async def advanced_rag_pipeline(inputs: ta.Bundle) -> str:
    """Advanced RAG pipeline with Bundle transformations."""
    # Step 1: Enrich query with user context
    query, user_context = await ta.unpack(inputs, "query", "user_context")
    enriched_query = await enrich_query(query, user_context)

    # Step 2: Update inputs with enriched query
    inputs_v2 = await ta.replace(inputs, query=enriched_query)

    # Step 3: Retrieve documents
    query, db = await ta.unpack(inputs_v2, "query", "db")
    docs = await retrieve_docs(query, db)

    # Step 4: Rerank documents
    (rerank_model,) = await ta.unpack(inputs, "rerank_model")
    reranked_docs = await rerank_docs(docs, rerank_model)

    # Step 5: Extend inputs with reranked docs
    generation_inputs = await ta.extend(inputs_v2, docs=reranked_docs)

    # Step 6: Generate answer
    query, docs, llm = await ta.unpack(generation_inputs, "query", "docs", "llm")
    answer = await generate_answer(query, docs, llm)

    return answer


# === Example 3: Type-Safe Inputs with make_inputs ===


# Define a typed input class
if not TYPE_CHECKING:
    RagInputs = ta.make_inputs(
        query=str,
        llm=str,
        db=str,
        temperature=float,
    )


@ta.jit
async def typed_pipeline(inputs: RagInputs) -> str:
    """Pipeline with type-safe inputs."""
    # Bundle operations work the same with typed inputs
    query, db, llm = await ta.unpack(inputs, "query", "db", "llm")
    docs = await retrieve_docs(query, db)
    answer = await generate_answer(query, docs, llm)

    return answer


# === Example 4: Compile-time vs Runtime Operations ===


@ta.jit
async def mixed_operations_pipeline(query: str, llm: str, db: str) -> str:
    """Demonstrates mixing compile-time and runtime Bundle operations."""
    # Compile-time: Create bundle from TracedValues
    # This happens during tracing, no graph node created
    big_bundle = ta.Bundle(
        query=query,
        llm=llm,
        db=db,
        config={"temperature": 0.7},
        metadata={"version": "1.0"},
    )

    # Runtime: Unpack fields directly
    query, db = await ta.unpack(big_bundle, "query", "db")
    docs = await retrieve_docs(query, db)

    # Runtime: Extend creates a graph node
    gen_bundle = await ta.extend(big_bundle, docs=docs)

    # Unpack all fields for final generation
    query, docs, llm = await ta.unpack(gen_bundle, "query", "docs", "llm")
    answer = await generate_answer(query, docs, llm)

    return answer


# === Main Demo ===


async def main():
    print("=" * 70)
    print("Example 1: Basic Bundle Usage")
    print("=" * 70)

    inputs = ta.Bundle(query="what is machine learning?", llm="gpt-4", db="vector_db")
    result = await basic_rag_pipeline(inputs)
    print(f"Result: {result}\n")

    print("=" * 70)
    print("Example 2: Advanced Pipeline with Bundle Transformations")
    print("=" * 70)

    advanced_inputs = ta.Bundle(
        query="what is deep learning?",
        llm="claude-3",
        db="vector_db",
        user_context={"name": "Alice", "role": "researcher"},
        rerank_model="cohere-rerank",
    )
    result = await advanced_rag_pipeline(advanced_inputs)
    print(f"Result: {result}\n")

    print("=" * 70)
    print("Example 3: Type-Safe Inputs")
    print("=" * 70)

    typed_inputs = RagInputs(
        query="explain neural networks",
        llm="gpt-4",
        db="arxiv_db",
        temperature=0.8,
    )
    result = await typed_pipeline(typed_inputs)
    print(f"Result: {result}\n")

    print("=" * 70)
    print("Example 4: Mixed Compile-time and Runtime Operations")
    print("=" * 70)

    result = await mixed_operations_pipeline(
        query="what is reinforcement learning?", llm="claude-3", db="papers_db"
    )
    print(f"Result: {result}\n")

    print("=" * 70)
    print("Bundle API Summary")
    print("=" * 70)
    print("""
Key Points:
- Bundle: Lightweight traced container for pipeline inputs
- ta.unpack(): Extract multiple fields as tuple (recommended!)
  * Clean, Pythonic syntax: query, db = await ta.unpack(inputs, "query", "db")
  * Works everywhere: @ta.jit, @ta.node, and regular async functions
- ta.get(): Get a single field
- ta.subset(): Extract fields as new Bundle (for passing around)
- ta.extend(): Add new fields (creates graph node)
- ta.replace(): Update existing fields (creates graph node)
- make_inputs(): Create type-safe input classes
    """)


if __name__ == "__main__":
    asyncio.run(main())
