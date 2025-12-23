"""Declarative ThinkAgain Demo."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import node, run


# User-defined state - framework doesn't impose structure
@dataclass
class RAGState:
    query: str
    documents: list[str] = field(default_factory=list)
    answer: str = ""
    quality: float = 0.0
    retrieval_attempt: int = 0
    refinements: int = 0
    top_n: int = 2


# Pure functions with @node decorator - no Context awareness needed
@node
async def retrieve_docs(s: RAGState) -> RAGState:
    attempt = s.retrieval_attempt + 1
    docs = [f"{s.query} fact #{i}" for i in range(1, min(4, 1 + attempt))]
    return RAGState(
        query=s.query,
        documents=docs,
        retrieval_attempt=attempt,
        refinements=s.refinements,
        top_n=s.top_n,
    )


@node
async def rerank_docs(s: RAGState) -> RAGState:
    return RAGState(
        query=s.query,
        documents=s.documents[: s.top_n],
        retrieval_attempt=s.retrieval_attempt,
        refinements=s.refinements,
        top_n=s.top_n,
    )


@node
async def generate_answer(s: RAGState) -> RAGState:
    docs = ", ".join(s.documents) if s.documents else "no docs"
    answer = f"Answer about '{s.query}' using {docs}"
    return RAGState(
        query=s.query,
        documents=s.documents,
        answer=answer,
        retrieval_attempt=s.retrieval_attempt,
        refinements=s.refinements,
        top_n=s.top_n,
    )


@node
async def critique_answer(s: RAGState) -> RAGState:
    quality = 0.9 if len(s.documents) >= 3 else 0.7 if len(s.documents) == 2 else 0.4
    return RAGState(
        query=s.query,
        documents=s.documents,
        answer=s.answer,
        quality=quality,
        retrieval_attempt=s.retrieval_attempt,
        refinements=s.refinements,
        top_n=s.top_n,
    )


@node
async def refine_query(s: RAGState) -> RAGState:
    r = s.refinements + 1
    return RAGState(
        query=f"{s.query} (refined {r})",
        documents=s.documents,
        answer=s.answer,
        quality=s.quality,
        retrieval_attempt=s.retrieval_attempt,
        refinements=r,
        top_n=s.top_n,
    )


# Demo 1: Simple pipeline - no async, no await!
def simple_rag(ctx):
    ctx = retrieve_docs(ctx)
    ctx = rerank_docs(ctx)
    ctx = generate_answer(ctx)
    return ctx


# Demo 2: Conditional - auto-materializes when accessing ctx.data
def conditional_pipeline(ctx):
    ctx = retrieve_docs(ctx)
    if len(ctx.data.documents) > 2:  # Accessing .data triggers materialization
        ctx = rerank_docs(ctx)
    ctx = generate_answer(ctx)
    return ctx


# Demo 3: Loop - auto-materializes each iteration
def self_correcting_rag(ctx):
    ctx = retrieve_docs(ctx)
    ctx = generate_answer(ctx)
    ctx = critique_answer(ctx)

    while ctx.data.quality < 0.8 and ctx.data.retrieval_attempt < 3:
        ctx = refine_query(ctx)
        ctx = retrieve_docs(ctx)
        ctx = generate_answer(ctx)
        ctx = critique_answer(ctx)

    return ctx


if __name__ == "__main__":
    print("1) Simple pipeline")
    result = run(simple_rag, RAGState(query="test", top_n=2))
    print(f"   {result.data.answer}\n")

    print("2) Conditional")
    result = run(conditional_pipeline, RAGState(query="test", retrieval_attempt=2))
    print(f"   {result.data.answer}\n")

    print("3) Self-correcting loop")
    result = run(self_correcting_rag, RAGState(query="test"))
    print(f"   {result.data.answer}")
    print(
        f"   quality={result.data.quality}, attempts={result.data.retrieval_attempt}",
    )
