"""Regression tests for execution-time cycles."""

import asyncio

from thinkagain import Context, Graph, END, Executable


class CounterExecutable(Executable):
    """Increment a counter stored in the Context."""

    async def arun(self, ctx: Context) -> Context:
        ctx.count = (ctx.count or 0) + 1
        return ctx


def _route(ctx: Context) -> str:
    """Loop until we've incremented three times."""
    return "worker" if (ctx.count or 0) < 3 else END


def _build_cycle_graph(name: str = "cycle_graph") -> Graph:
    graph = Graph(name=name, max_steps=10)
    graph.add("worker", CounterExecutable())
    graph.set_entry("worker")
    graph.edge("worker", _route)
    return graph


def _build_nested_graph() -> Graph:
    inner = _build_cycle_graph(name="inner_cycle")
    outer = Graph(name="outer")
    outer.add("subgraph", inner)
    outer.set_entry("subgraph")
    outer.edge("subgraph", END)
    return outer


def _run(compiled) -> Context:
    return asyncio.run(compiled.arun(Context()))


def _assert_three_iterations(ctx: Context) -> None:
    assert ctx.count == 3
    assert ctx.execution_path == ["worker", "worker", "worker"]
    assert ctx.total_steps == 3


def test_compiled_cycle() -> None:
    compiled = _build_cycle_graph().compile()
    ctx = _run(compiled)
    _assert_three_iterations(ctx)


def test_nested_subgraph_with_cycle() -> None:
    compiled = _build_nested_graph().compile()
    ctx = _run(compiled)
    assert ctx.count == 3
