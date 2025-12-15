"""Declarative ThinkAgain Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import node, run


# Plain async functions with @node decorator
@node
async def retrieve_docs(ctx):
    attempt = ctx.get("retrieval_attempt", 0) + 1
    ctx.retrieval_attempt = attempt
    ctx.documents = [f"{ctx.query} fact #{i}" for i in range(1, min(4, 1 + attempt))]
    return ctx


@node
async def rerank_docs(ctx):
    ctx.documents = ctx.documents[: ctx.get("top_n", 2)]
    return ctx


@node
async def generate_answer(ctx):
    docs = ", ".join(ctx.documents) if ctx.documents else "no docs"
    ctx.answer = f"Answer about '{ctx.query}' using {docs}"
    return ctx


@node
async def critique_answer(ctx):
    ctx.quality = 0.9 if len(ctx.documents) >= 3 else 0.7 if len(ctx.documents) == 2 else 0.4
    return ctx


@node
async def refine_query(ctx):
    r = ctx.get("refinements", 0) + 1
    ctx.refinements = r
    ctx.query = f"{ctx.query} (refined {r})"
    return ctx


# Demo 1: Simple pipeline - no async, no await!
def simple_rag(ctx):
    ctx = retrieve_docs(ctx)
    ctx = rerank_docs(ctx)
    ctx = generate_answer(ctx)
    return ctx


# Demo 2: Conditional - auto-materializes when accessing ctx.documents
def conditional_pipeline(ctx):
    ctx = retrieve_docs(ctx)
    if len(ctx.documents) > 2:  # auto-materializes here
        ctx = rerank_docs(ctx)
    ctx = generate_answer(ctx)
    return ctx


# Demo 3: Loop - auto-materializes each iteration
def self_correcting_rag(ctx):
    ctx = retrieve_docs(ctx)
    ctx = generate_answer(ctx)
    ctx = critique_answer(ctx)

    while ctx.quality < 0.8 and ctx.retrieval_attempt < 3:  # auto-materializes
        ctx = refine_query(ctx)
        ctx = retrieve_docs(ctx)
        ctx = generate_answer(ctx)
        ctx = critique_answer(ctx)

    return ctx


if __name__ == "__main__":
    print("1) Simple pipeline")
    result = run(simple_rag, {"query": "test", "top_n": 2})
    print(f"   {result.answer}\n")

    print("2) Conditional")
    result = run(conditional_pipeline, {"query": "test", "retrieval_attempt": 2})
    print(f"   {result.answer}\n")

    print("3) Self-correcting loop")
    result = run(self_correcting_rag, {"query": "test"})
    print(f"   {result.answer}")
    print(f"   quality={result.quality}, attempts={result.retrieval_attempt}")
