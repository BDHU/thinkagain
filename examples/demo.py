"""Improved API demo.

Highlights:
- Context[T] type hints
- Sequential pipelines
- dataclasses.replace() to reduce boilerplate
"""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Context, Sequential, node


@dataclass
class RAGState:
    query: str
    documents: list[str] = field(default_factory=list)
    answer: str = ""
    quality: float = 0.0
    retrieval_attempt: int = 0
    refinements: int = 0
    top_n: int = 2


@node
async def retrieve_docs(s: RAGState) -> RAGState:
    attempt = s.retrieval_attempt + 1
    docs = [f"{s.query} fact #{i}" for i in range(1, min(4, 1 + attempt))]
    return replace(
        s,
        documents=docs,
        retrieval_attempt=attempt,
    )


@node
async def rerank_docs(s: RAGState) -> RAGState:
    return replace(s, documents=s.documents[: s.top_n])


@node
async def generate_answer(s: RAGState) -> RAGState:
    docs = ", ".join(s.documents) if s.documents else "no docs"
    answer = f"Answer about '{s.query}' using {docs}"
    return replace(s, answer=answer)


@node
async def critique_answer(s: RAGState) -> RAGState:
    quality = 0.9 if len(s.documents) >= 3 else 0.7 if len(s.documents) == 2 else 0.4
    return replace(s, quality=quality)


@node
async def refine_query(s: RAGState) -> RAGState:
    r = s.refinements + 1
    return replace(
        s,
        query=f"{s.query} (refined {r})",
        refinements=r,
    )


simple_rag = Sequential(
    retrieve_docs,
    rerank_docs,
    generate_answer,
)

full_pipeline = Sequential(
    simple_rag,
    critique_answer,
)


def conditional_pipeline(ctx: Context[RAGState]) -> Context[RAGState]:
    ctx = retrieve_docs(ctx)
    if len(ctx.data.documents) > 2:
        ctx = rerank_docs(ctx)

    ctx = generate_answer(ctx)
    return ctx


def self_correcting_rag(ctx: Context[RAGState]) -> Context[RAGState]:
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
    print("ThinkAgain demo")
    print(f"Pipeline: {simple_rag}")
    ctx = simple_rag(Context(RAGState(query="test", top_n=2)))
    print(f"Sequential: {ctx.data.answer}")

    ctx = conditional_pipeline(Context(RAGState(query="test", retrieval_attempt=2)))
    print(f"Conditional: {ctx.data.answer}")

    ctx = self_correcting_rag(Context(RAGState(query="test")))
    print(f"Self-correcting: {ctx.data.answer}")
    print(f"Quality/Attempts: {ctx.data.quality}/{ctx.data.retrieval_attempt}")

    ctx = full_pipeline(Context(RAGState(query="composed")))
    print(f"Composed: {ctx.data.answer} (quality {ctx.data.quality})")
    print(f"Nodes: {full_pipeline.node_names}")
