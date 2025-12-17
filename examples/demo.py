"""Declarative ThinkAgain Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import node, run


# Plain async functions with @node decorator
@node
async def retrieve_docs(ctx):
    attempt = ctx.get("retrieval_attempt", 0) + 1
    ctx.set("retrieval_attempt", attempt)
    query = ctx.get("query")
    ctx.set(
        "documents",
        [f"{query} fact #{i}" for i in range(1, min(4, 1 + attempt))],
    )
    return ctx


@node
async def rerank_docs(ctx):
    docs = ctx.get("documents", [])
    top_n = ctx.get("top_n", 2)
    ctx.set("documents", docs[:top_n])
    return ctx


@node
async def generate_answer(ctx):
    docs_list = ctx.get("documents", [])
    docs = ", ".join(docs_list) if docs_list else "no docs"
    query = ctx.get("query")
    ctx.set("answer", f"Answer about '{query}' using {docs}")
    return ctx


@node
async def critique_answer(ctx):
    docs = ctx.get("documents", [])
    quality = 0.9 if len(docs) >= 3 else 0.7 if len(docs) == 2 else 0.4
    ctx.set("quality", quality)
    return ctx


@node
async def refine_query(ctx):
    r = ctx.get("refinements", 0) + 1
    ctx.set("refinements", r)
    query = ctx.get("query")
    ctx.set("query", f"{query} (refined {r})")
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
    if len(ctx.get("documents", [])) > 2:
        ctx = rerank_docs(ctx)
    ctx = generate_answer(ctx)
    return ctx


# Demo 3: Loop - auto-materializes each iteration
def self_correcting_rag(ctx):
    ctx = retrieve_docs(ctx)
    ctx = generate_answer(ctx)
    ctx = critique_answer(ctx)

    while ctx.get("quality") < 0.8 and ctx.get("retrieval_attempt") < 3:
        ctx = refine_query(ctx)
        ctx = retrieve_docs(ctx)
        ctx = generate_answer(ctx)
        ctx = critique_answer(ctx)

    return ctx


if __name__ == "__main__":
    print("1) Simple pipeline")
    result = run(simple_rag, {"query": "test", "top_n": 2})
    print(f"   {result.get('answer')}\n")

    print("2) Conditional")
    result = run(conditional_pipeline, {"query": "test", "retrieval_attempt": 2})
    print(f"   {result.get('answer')}\n")

    print("3) Self-correcting loop")
    result = run(self_correcting_rag, {"query": "test"})
    print(f"   {result.get('answer')}")
    print(
        f"   quality={result.get('quality')}, "
        f"attempts={result.get('retrieval_attempt')}",
    )
