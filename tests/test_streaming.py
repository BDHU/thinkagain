from __future__ import annotations

import asyncio

from thinkagain import Context, Graph, Worker, END, CompiledGraph


class Appender(Worker):
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
) -> tuple[Context, list[tuple[str, str | None, list[str]]]]:
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


# ============================================================================
# Tests for Worker.astream() incremental streaming
# ============================================================================


class StreamingCounter(Worker):
    """Worker that yields 3 incremental updates."""

    async def astream(self, ctx: Context):
        for i in range(1, 4):
            ctx.count = i
            yield ctx


def test_worker_astream_yields_incremental_updates():
    """Test streaming worker yields multiple snapshots."""

    async def run():
        worker = StreamingCounter()
        ctx = Context()
        counts = [c.count async for c in worker.astream(ctx)]
        assert counts == [1, 2, 3]

    asyncio.run(run())


def test_graph_streaming_flag():
    """Test graph.stream() yields final result from streaming worker.

    Note: The parallel execution model doesn't yield intermediate streaming
    events - it only yields the final result from each node. For true streaming
    of intermediate results, use Worker.astream() directly.
    """

    async def run():
        graph = Graph(name="test")
        graph.add("counter", StreamingCounter())
        graph.edge("counter", END)

        compiled = graph.compile()
        final_count = None

        async for event in compiled.stream(Context()):
            if event.type == "node" and hasattr(event.ctx, "count"):
                final_count = event.ctx.count

        assert final_count == 3

    asyncio.run(run())
