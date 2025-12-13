"""Regression tests for execution-time cycles."""

import asyncio

from thinkagain.core.context import Context
from thinkagain.core.graph import END, Graph
from thinkagain.core.worker import Worker


class CounterWorker(Worker):
    """Increment a counter stored in the Context."""

    async def arun(self, ctx: Context) -> Context:
        ctx.count = (ctx.count or 0) + 1
        return ctx


def _route(ctx: Context) -> str:
    """Loop until we've incremented three times."""
    return "loop" if (ctx.count or 0) < 3 else "done"


def _build_cycle_graph(name: str = "cycle_graph") -> Graph:
    graph = Graph(name=name, max_steps=10)
    graph.add_node("worker", CounterWorker())
    graph.set_entry("worker")
    graph.add_conditional_edge(
        "worker",
        route=_route,
        paths={"loop": "worker", "done": END},
    )
    return graph


def _build_nested_graph() -> Graph:
    inner = _build_cycle_graph(name="inner_cycle")
    outer = Graph(name="outer")
    outer.add_node("subgraph", inner)
    outer.set_entry("subgraph")
    outer.add_edge("subgraph", END)
    return outer


def _run(executable) -> Context:
    """Helper to execute a graph/compiled graph and return its context."""
    return asyncio.run(executable.arun(Context()))


def _assert_three_iterations(ctx: Context) -> None:
    assert ctx.count == 3
    assert ctx.execution_path == ["worker", "worker", "worker"]
    assert ctx.total_steps == 3


def test_non_compiled_cycle() -> None:
    ctx = _run(_build_cycle_graph())
    _assert_three_iterations(ctx)


def test_compiled_cycle() -> None:
    compiled = _build_cycle_graph().compile()
    ctx = _run(compiled)
    _assert_three_iterations(ctx)


def test_nested_subgraph_with_cycle() -> None:
    compiled = _build_nested_graph().compile()
    ctx = _run(compiled)
    assert ctx.count == 3
