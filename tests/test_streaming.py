from __future__ import annotations

import asyncio

from typing import Optional

from thinkagain import Context, Graph, Executable, END, CompiledGraph


class Appender(Executable):
    def __init__(self, label: str):
        super().__init__(name=f"append_{label}")
        self.label = label

    async def arun(self, ctx: Context) -> Context:
        ctx.values = ctx.get("values", []) + [self.label]
        return ctx


def _build_linear_graph() -> Graph:
    graph = Graph(name="linear")
    graph.add("first", Appender("a"))
    graph.add("second", Appender("b"))
    graph.set_entry("first")
    graph.edge("first", "second")
    graph.edge("second", END)
    return graph


async def _gather_events(
    compiled: CompiledGraph,
) -> tuple[Context, list[tuple[str, Optional[str], list[str]]]]:
    ctx = Context()
    events = []
    final_ctx = ctx
    async for event in compiled.stream(ctx):
        # Only collect final node events (not intermediate streaming events)
        if event.type != "node" or not event.streaming:
            events.append((event.type, event.node, list(event.ctx.get("values", []))))
        if event.type == "end":
            final_ctx = event.ctx
    return final_ctx, events


def test_graph_stream_emits_ordered_events():
    compiled = _build_linear_graph().compile()
    ctx, events = asyncio.run(_gather_events(compiled))
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


class StreamingCounter(Executable):
    """Executable that yields 3 incremental updates."""

    async def astream(self, ctx: Context):
        for i in range(1, 4):
            ctx.count = i
            yield ctx


def test_worker_astream_yields_incremental_updates():
    async def run():
        worker = StreamingCounter()
        ctx = Context()
        counts = [c.count async for c in worker.astream(ctx)]
        assert counts == [1, 2, 3]

    asyncio.run(run())


def test_graph_streaming_flag():
    """Streaming executables yield intermediate events (streaming=True)."""

    async def run():
        graph = Graph(name="test")
        graph.add("counter", StreamingCounter())
        graph.edge("counter", END)

        compiled = graph.compile()
        streaming_counts = []
        final_count = None

        async for event in compiled.stream(Context()):
            if event.type == "node":
                if event.streaming:
                    streaming_counts.append(event.ctx.count)
                else:
                    final_count = event.ctx.count

        assert streaming_counts == [1, 2, 3], (
            f"Expected [1, 2, 3], got {streaming_counts}"
        )
        assert final_count == 3

    asyncio.run(run())
