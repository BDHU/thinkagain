from __future__ import annotations

import asyncio

from thinkagain import Context, Executable, Graph, Worker, END


class Appender(Worker):
    def __init__(self, label: str):
        super().__init__(name=f"append_{label}")
        self.label = label

    def __call__(self, ctx: Context) -> Context:
        ctx.values = ctx.get("values", []) + [self.label]
        return ctx


def _build_linear_graph() -> Graph:
    graph = Graph(name="linear")
    graph.add_node("first", Appender("a"))
    graph.add_node("second", Appender("b"))
    graph.set_entry("first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph


async def _gather_events(
    executable: "Executable",
) -> tuple[Context, list[tuple[str, str | None, list[str]]]]:
    ctx = Context()
    events = []
    async for event in executable.stream(ctx):
        events.append((event.type, event.node, list(ctx.get("values", []))))
    return ctx, events


def test_graph_stream_emits_ordered_events():
    graph = _build_linear_graph()
    ctx, events = asyncio.run(_gather_events(graph))
    assert [e[0] for e in events] == ["start", "node", "node", "end"]
    assert [e[1] for e in events if e[0] == "node"] == ["first", "second"]
    assert ctx.values == ["a", "b"]
    assert ctx.execution_path == ["first", "second"]


def test_compiled_graph_stream_matches_runtime():
    compiled = _build_linear_graph().compile()
    ctx, events = asyncio.run(_gather_events(compiled))
    assert events[0][1] == "first"
    assert events[-1][0] == "end"
    assert ctx.values == ["a", "b"]
