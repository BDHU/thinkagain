"""
Minimal ThinkAgain Demo
=======================

Demonstrates the core primitives:
1. Simple graph with linear flow
2. Graphs with conditional routing and cycles
3. Composing subgraphs and compiling them

Run with: ``python examples/minimal_demo.py``
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Executable, Graph, END


# -----------------------------------------------------------------------------
# Basic executables
# -----------------------------------------------------------------------------


class RetrieveDocs(Executable):
    """Simulate document retrieval."""

    async def arun(self, ctx: Context) -> Context:
        attempt = ctx.get("retrieval_attempt", 0) + 1
        ctx.retrieval_attempt = attempt
        doc_count = min(3, 1 + attempt)
        ctx.documents = [f"{ctx.query} fact #{i}" for i in range(1, doc_count + 1)]
        ctx.log(
            f"[{self.name}] Retrieved {len(ctx.documents)} docs (attempt {attempt})"
        )
        return ctx


class RerankDocs(Executable):
    """Keep only top documents."""

    async def arun(self, ctx: Context) -> Context:
        keep = ctx.get("top_n", 2)
        ctx.documents = ctx.documents[:keep]
        ctx.log(f"[{self.name}] Keeping top {len(ctx.documents)} docs")
        return ctx


class GenerateAnswer(Executable):
    """Pretend to call an LLM."""

    async def arun(self, ctx: Context) -> Context:
        doc_summary = ", ".join(ctx.documents) if ctx.documents else "no docs"
        ctx.answer = f"Answer about '{ctx.query}' using {doc_summary}"
        ctx.log(f"[{self.name}] Generated answer with {len(ctx.documents)} docs")
        return ctx


class CritiqueAnswer(Executable):
    """Provide a quality score."""

    async def arun(self, ctx: Context) -> Context:
        doc_count = len(ctx.documents)
        ctx.quality = 0.9 if doc_count >= 3 else 0.7 if doc_count == 2 else 0.4
        ctx.log(f"[{self.name}] Quality score = {ctx.quality:.2f}")
        return ctx


class RefineQuery(Executable):
    """Refine the query when critique is not satisfied."""

    async def arun(self, ctx: Context) -> Context:
        refinements = ctx.get("refinements", 0) + 1
        ctx.refinements = refinements
        ctx.query = f"{ctx.query} (detail {refinements})"
        ctx.log(f"[{self.name}] Refining query to '{ctx.query}'")
        return ctx


# -----------------------------------------------------------------------------
# Demo 1: Simple linear graph
# -----------------------------------------------------------------------------


def simple_graph_demo() -> None:
    print("\n" + "=" * 72)
    print("1) Simple linear graph")
    print("=" * 72)

    graph = Graph(name="rag_pipeline")
    graph.add("retrieve", RetrieveDocs())
    graph.add("rerank", RerankDocs())
    graph.add("generate", GenerateAnswer())
    graph.edge("retrieve", "rerank")
    graph.edge("rerank", "generate")
    graph.edge("generate", END)

    ctx = Context(query="thinkagain overview", top_n=2)
    result = asyncio.run(graph.compile().arun(ctx))

    print(f"Answer: {result.answer}")
    print(f"Execution path: {' -> '.join(result.execution_path)}")


# -----------------------------------------------------------------------------
# Demo 2: Graphs with conditional routing
# -----------------------------------------------------------------------------


def build_self_correcting_graph() -> Graph:
    graph = Graph(name="self_correcting_rag", max_steps=20)
    graph.add("retrieve", RetrieveDocs())
    graph.add("generate", GenerateAnswer())
    graph.add("critique", CritiqueAnswer())
    graph.add("refine", RefineQuery())

    graph.set_entry("retrieve")
    graph.edge("retrieve", "generate")
    graph.edge("generate", "critique")

    def routing(ctx: Context) -> str:
        high_quality = ctx.quality >= 0.8
        max_attempts = ctx.retrieval_attempt >= 3
        return END if high_quality or max_attempts else "refine"

    graph.edge("critique", routing)
    graph.edge("refine", "retrieve")
    return graph


async def graph_demo() -> None:
    print("\n" + "=" * 72)
    print("2) Graph with conditional routing and cycles")
    print("=" * 72)

    graph = build_self_correcting_graph()
    ctx = Context(query="why explicit graphs matter", top_n=2)
    result = await graph.compile().arun(ctx)

    print(f"Answer: {result.answer}")
    print(f"Quality: {result.quality:.2f} after {result.retrieval_attempt} attempts")
    print(f"Execution path: {' -> '.join(result.execution_path)}")
    print("\nGraph structure (Mermaid):")
    print(graph.visualize())


# -----------------------------------------------------------------------------
# Demo 3: Composition + compile()
# -----------------------------------------------------------------------------


def build_retrieval_stage() -> Graph:
    graph = Graph(name="retrieval_stage")
    graph.add("retrieve", RetrieveDocs())
    graph.add("rerank", RerankDocs())
    graph.edge("retrieve", "rerank")
    graph.edge("rerank", END)
    return graph


async def composition_demo() -> None:
    print("\n" + "=" * 72)
    print("3) Composing subgraphs")
    print("=" * 72)

    # Build outer graph with subgraph
    outer = Graph(name="composed_pipeline")
    outer.add("retrieval", build_retrieval_stage())
    outer.add("generate", GenerateAnswer())
    outer.edge("retrieval", "generate")
    outer.edge("generate", END)

    compiled = outer.compile()
    ctx = Context(query="subgraph composition", top_n=2)
    result = await compiled.arun(ctx)

    print(f"Answer: {result.answer}")
    print(f"Execution path: {' -> '.join(result.execution_path)}")
    print(f"Flattened nodes: {list(compiled.nodes.keys())}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def async_main() -> None:
    await graph_demo()
    await composition_demo()


if __name__ == "__main__":
    simple_graph_demo()
    asyncio.run(async_main())
