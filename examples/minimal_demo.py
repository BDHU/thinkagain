"""
Minimal ThinkAgain Demo
=======================

This single script keeps the examples approachable while still covering all
of the core primitives provided by thinkagain:

1. Sequential pipelines composed with the ``>>`` operator
2. Explicit graphs with conditional routing and cycles
3. Composing subgraphs together and compiling them into an executable plan

Run with: ``python examples/minimal_demo.py``
"""

import asyncio
import sys
from pathlib import Path

# Add project root to the import path when the file is executed directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Worker, Graph, END


# -----------------------------------------------------------------------------
# Basic workers reused across the scenarios below
# -----------------------------------------------------------------------------


class RetrieveDocs(Worker):
    """Simulate document retrieval from a knowledge source."""

    async def arun(self, ctx: Context) -> Context:
        attempt = ctx.get("retrieval_attempt", 0) + 1
        ctx.retrieval_attempt = attempt

        # Slowly increase recall on each retry to illustrate graph loops
        doc_count = min(3, 1 + attempt)
        ctx.documents = [f"{ctx.query} fact #{i}" for i in range(1, doc_count + 1)]
        ctx.log(
            f"[{self.name}] Retrieved {len(ctx.documents)} docs (attempt {attempt})"
        )
        return ctx


class RerankDocs(Worker):
    """Keep only the top documents so downstream workers stay simple."""

    async def arun(self, ctx: Context) -> Context:
        keep = ctx.get("top_n", 2)
        ctx.documents = ctx.documents[:keep]
        ctx.log(f"[{self.name}] Keeping top {len(ctx.documents)} docs")
        return ctx


class GenerateAnswer(Worker):
    """Pretend to call an LLM to produce an answer."""

    async def arun(self, ctx: Context) -> Context:
        doc_summary = ", ".join(ctx.documents) if ctx.documents else "no docs"
        ctx.answer = f"Answer about '{ctx.query}' using {doc_summary}"
        ctx.log(f"[{self.name}] Generated answer with {len(ctx.documents)} docs")
        return ctx


class CritiqueAnswer(Worker):
    """Provide a toy quality score so the graph can branch."""

    async def arun(self, ctx: Context) -> Context:
        doc_count = len(ctx.documents)
        ctx.quality = 0.9 if doc_count >= 3 else 0.7 if doc_count == 2 else 0.4
        ctx.log(f"[{self.name}] Quality score = {ctx.quality:.2f}")
        return ctx


class RefineQuery(Worker):
    """Refine the query when the critique is not satisfied."""

    async def arun(self, ctx: Context) -> Context:
        refinements = ctx.get("refinements", 0) + 1
        ctx.refinements = refinements
        ctx.query = f"{ctx.query} (detail {refinements})"
        ctx.log(f"[{self.name}] Refining query to '{ctx.query}'")
        return ctx


# -----------------------------------------------------------------------------
# Demo 1: Sequential pipelines
# -----------------------------------------------------------------------------


def sequential_pipeline_demo() -> None:
    print("\n" + "=" * 72)
    print("1) Sequential pipelines with >> operator")
    print("=" * 72)

    # Build the pipeline using the >> operator (sugar over Graph)
    rag_pipeline = RetrieveDocs() >> RerankDocs() >> GenerateAnswer()
    async_ctx = Context(query="thinkagain overview", top_n=2)
    async_result = asyncio.run(rag_pipeline.arun(async_ctx))

    print(f"Async answer: {async_result.answer}")
    print(f"Execution path: {' → '.join(async_result.execution_path)}")

    # Sync execution also works
    sync_pipeline = RetrieveDocs() >> RerankDocs() >> GenerateAnswer()
    sync_result = sync_pipeline(Context(query="sync usage", top_n=1))

    print(f"Sync answer:  {sync_result.answer}")
    print(f"History tail: {sync_result.history[-2:]}")


# -----------------------------------------------------------------------------
# Demo 2: Graphs with conditional routing
# -----------------------------------------------------------------------------


def build_self_correcting_graph() -> Graph:
    graph = Graph(name="self_correcting_rag")
    graph.add_node("retrieve", RetrieveDocs())
    graph.add_node("generate", GenerateAnswer())
    graph.add_node("critique", CritiqueAnswer())
    graph.add_node("refine", RefineQuery())

    graph.set_entry("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")

    def routing(ctx: Context) -> str:
        high_quality = ctx.quality >= 0.8
        max_attempts = ctx.retrieval_attempt >= 3
        return "done" if high_quality or max_attempts else "refine"

    graph.add_conditional_edge(
        "critique",
        route=routing,
        paths={
            "done": END,
            "refine": "refine",
        },
    )
    graph.add_edge("refine", "retrieve")  # Cycle back to retrieval
    return graph


async def graph_demo() -> None:
    print("\n" + "=" * 72)
    print("2) Explicit graphs with a feedback loop")
    print("=" * 72)

    agent = build_self_correcting_graph()
    ctx = Context(query="why explicit graphs matter", top_n=2)
    result = await agent.arun(ctx)

    print(f"Answer: {result.answer}")
    print(
        f"Quality: {result.quality:.2f} after {result.retrieval_attempt} retrieval attempts"
    )
    print(f"Execution path: {' → '.join(result.execution_path)}")
    print("\nGraph structure (Mermaid):")
    print(agent.visualize())


# -----------------------------------------------------------------------------
# Demo 3: Composition + compile()
# -----------------------------------------------------------------------------


def build_retrieval_stage() -> Graph:
    return RetrieveDocs() >> RerankDocs()


def build_generation_stage() -> Graph:
    return GenerateAnswer()


async def composition_and_compile_demo() -> None:
    print("\n" + "=" * 72)
    print("3) Composing subgraphs and compiling the plan")
    print("=" * 72)

    retrieval = build_retrieval_stage()
    generation = build_generation_stage()

    composed = retrieval >> generation
    compiled = composed.compile()

    ctx = Context(query="compilation benefits", top_n=2)
    result = await compiled.arun(ctx)

    print(f"Compiled answer: {result.answer}")
    print(f"Flat execution path: {' → '.join(result.execution_path)}")
    print(f"Recent logs: {result.history[-3:]}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


async def async_main() -> None:
    await graph_demo()
    await composition_and_compile_demo()


if __name__ == "__main__":
    sequential_pipeline_demo()
    asyncio.run(async_main())
