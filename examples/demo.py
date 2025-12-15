"""Declarative ThinkAgain Demo."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkagain import Node, run


# Plain async functions - no base class needed
async def retrieve_docs(ctx):
    attempt = ctx.get("retrieval_attempt", 0) + 1
    ctx.retrieval_attempt = attempt
    ctx.documents = [f"{ctx.query} fact #{i}" for i in range(1, min(4, 1 + attempt))]
    return ctx


async def rerank_docs(ctx):
    ctx.documents = ctx.documents[: ctx.get("top_n", 2)]
    return ctx


async def generate_answer(ctx):
    docs = ", ".join(ctx.documents) if ctx.documents else "no docs"
    ctx.answer = f"Answer about '{ctx.query}' using {docs}"
    return ctx


async def critique_answer(ctx):
    ctx.quality = 0.9 if len(ctx.documents) >= 3 else 0.7 if len(ctx.documents) == 2 else 0.4
    return ctx


async def refine_query(ctx):
    r = ctx.get("refinements", 0) + 1
    ctx.refinements = r
    ctx.query = f"{ctx.query} (refined {r})"
    return ctx


# Wrap as nodes
retrieve = Node(retrieve_docs)
rerank = Node(rerank_docs)
generate = Node(generate_answer)
critique = Node(critique_answer)
refine = Node(refine_query)


# Demo 1: Simple pipeline
async def simple_rag(ctx):
    ctx = await retrieve(ctx)
    ctx = await rerank(ctx)
    ctx = await generate(ctx)
    return ctx


# Demo 2: Conditional
async def conditional_pipeline(ctx):
    ctx = await retrieve(ctx)
    ctx = await ctx  # materialize to check
    if len(ctx.documents) > 2:
        ctx = await rerank(ctx)
    ctx = await generate(ctx)
    return ctx


# Demo 3: Loop
async def self_correcting_rag(ctx):
    ctx = await retrieve(ctx)
    ctx = await generate(ctx)
    ctx = await critique(ctx)
    ctx = await ctx

    while ctx.quality < 0.8 and ctx.retrieval_attempt < 3:
        ctx = await refine(ctx)
        ctx = await retrieve(ctx)
        ctx = await generate(ctx)
        ctx = await critique(ctx)
        ctx = await ctx

    return ctx


if __name__ == "__main__":
    print("1) Simple pipeline")
    result = asyncio.run(run(simple_rag, {"query": "test", "top_n": 2}))
    print(f"   {result.answer}\n")

    print("2) Conditional")
    result = asyncio.run(run(conditional_pipeline, {"query": "test", "retrieval_attempt": 2}))
    print(f"   {result.answer}\n")

    print("3) Self-correcting loop")
    result = asyncio.run(run(self_correcting_rag, {"query": "test"}))
    print(f"   {result.answer}")
    print(f"   quality={result.quality}, attempts={result.retrieval_attempt}")
