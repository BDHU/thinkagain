"""
Declarative ThinkAgain Demo
============================

Demonstrates the declarative API with lazy materialization:
1. Simple linear pipeline
2. Conditional branching with if/else
3. Loops with while
4. Self-correcting RAG pattern

Run with: ``python examples/demo.py``
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Node, run, Context, Executable


# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------


class RetrieveDocs(Executable):
    """Simulate document retrieval."""

    async def arun(self, ctx: Context) -> Context:
        attempt = ctx.get("retrieval_attempt", 0) + 1
        ctx.retrieval_attempt = attempt
        doc_count = min(3, 1 + attempt)
        ctx.documents = [f"{ctx.query} fact #{i}" for i in range(1, doc_count + 1)]
        ctx.log(f"[retrieve] Got {len(ctx.documents)} docs (attempt {attempt})")
        return ctx


class RerankDocs(Executable):
    """Keep only top documents."""

    async def arun(self, ctx: Context) -> Context:
        keep = ctx.get("top_n", 2)
        ctx.documents = ctx.documents[:keep]
        ctx.log(f"[rerank] Keeping top {len(ctx.documents)} docs")
        return ctx


class GenerateAnswer(Executable):
    """Pretend to call an LLM."""

    async def arun(self, ctx: Context) -> Context:
        doc_summary = ", ".join(ctx.documents) if ctx.documents else "no docs"
        ctx.answer = f"Answer about '{ctx.query}' using {doc_summary}"
        ctx.log(f"[generate] Generated answer")
        return ctx


class CritiqueAnswer(Executable):
    """Provide a quality score."""

    async def arun(self, ctx: Context) -> Context:
        doc_count = len(ctx.documents)
        ctx.quality = 0.9 if doc_count >= 3 else 0.7 if doc_count == 2 else 0.4
        ctx.log(f"[critique] Quality = {ctx.quality:.2f}")
        return ctx


class RefineQuery(Executable):
    """Refine the query when critique is not satisfied."""

    async def arun(self, ctx: Context) -> Context:
        refinements = ctx.get("refinements", 0) + 1
        ctx.refinements = refinements
        ctx.query = f"{ctx.query} (refined {refinements})"
        ctx.log(f"[refine] New query: '{ctx.query}'")
        return ctx


# -----------------------------------------------------------------------------
# Declare nodes (once, at module level)
# -----------------------------------------------------------------------------

retrieve = Node(RetrieveDocs())
rerank = Node(RerankDocs())
generate = Node(GenerateAnswer())
critique = Node(CritiqueAnswer())
refine = Node(RefineQuery())


# -----------------------------------------------------------------------------
# Demo 1: Simple linear pipeline
# -----------------------------------------------------------------------------


async def simple_rag(ctx):
    """Simple RAG: retrieve -> rerank -> generate."""
    ctx = await retrieve(ctx)
    ctx = await rerank(ctx)
    ctx = await generate(ctx)
    return ctx


def demo_simple():
    print("\n" + "=" * 72)
    print("1) Simple linear pipeline (declarative)")
    print("=" * 72)

    result = asyncio.run(run(simple_rag, {"query": "thinkagain overview", "top_n": 2}))

    print(f"Answer: {result.answer}")
    print(f"History: {result.history}")


# -----------------------------------------------------------------------------
# Demo 2: Conditional branching
# -----------------------------------------------------------------------------


async def conditional_pipeline(ctx):
    """Branch based on input flag."""
    ctx = await retrieve(ctx)
    ctx = await ctx  # materialize to check documents

    if len(ctx.documents) > 2:
        print("  -> Taking full path (many docs)")
        ctx = await rerank(ctx)
        ctx = await generate(ctx)
    else:
        print("  -> Taking short path (few docs)")
        ctx = await generate(ctx)

    return ctx


def demo_conditional():
    print("\n" + "=" * 72)
    print("2) Conditional branching")
    print("=" * 72)

    # First attempt gets 2 docs -> short path
    print("\nWith retrieval_attempt=1:")
    result = asyncio.run(
        run(conditional_pipeline, {"query": "conditional test", "retrieval_attempt": 1})
    )
    print(f"Answer: {result.answer}")

    # Third attempt gets 3 docs -> full path
    print("\nWith retrieval_attempt=2:")
    result = asyncio.run(
        run(conditional_pipeline, {"query": "conditional test", "retrieval_attempt": 2})
    )
    print(f"Answer: {result.answer}")


# -----------------------------------------------------------------------------
# Demo 3: Self-correcting RAG with loop
# -----------------------------------------------------------------------------


async def self_correcting_rag(ctx):
    """Loop until quality is good enough or max attempts reached."""
    ctx = await retrieve(ctx)
    ctx = await generate(ctx)
    ctx = await critique(ctx)
    ctx = await ctx  # materialize to check quality

    while ctx.quality < 0.8 and ctx.retrieval_attempt < 3:
        print(f"  -> Quality {ctx.quality:.2f} too low, refining...")
        ctx = await refine(ctx)
        ctx = await retrieve(ctx)
        ctx = await generate(ctx)
        ctx = await critique(ctx)
        ctx = await ctx  # materialize for next check

    return ctx


def demo_self_correcting():
    print("\n" + "=" * 72)
    print("3) Self-correcting RAG with loop")
    print("=" * 72)

    result = asyncio.run(
        run(self_correcting_rag, {"query": "why declarative graphs matter", "top_n": 2})
    )

    print(f"\nFinal answer: {result.answer}")
    print(f"Final quality: {result.quality:.2f}")
    print(f"Attempts: {result.retrieval_attempt}")
    print(f"History: {result.history}")


# -----------------------------------------------------------------------------
# Demo 4: Nested helper functions
# -----------------------------------------------------------------------------


async def retrieve_and_rerank(ctx):
    """Helper: retrieve then rerank."""
    ctx = await retrieve(ctx)
    ctx = await rerank(ctx)
    return ctx


async def composed_pipeline(ctx):
    """Use helper functions for composition."""
    ctx = await retrieve_and_rerank(ctx)
    ctx = await generate(ctx)
    return ctx


def demo_composition():
    print("\n" + "=" * 72)
    print("4) Composition with helper functions")
    print("=" * 72)

    result = asyncio.run(run(composed_pipeline, {"query": "composition demo", "top_n": 2}))

    print(f"Answer: {result.answer}")
    print(f"History: {result.history}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    demo_simple()
    demo_conditional()
    demo_self_correcting()
    demo_composition()
