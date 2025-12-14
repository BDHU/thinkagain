"""
Runtime execution engine for compiled graphs.
Supports fan-out (parallel), cycles, and conditional routing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable as AwaitableABC
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

from .context import Context

EdgeTarget = Union[str, List[str], Callable[[Context], Union[str, List[str]]]]
EdgeMap = Dict[str, EdgeTarget]


@dataclass
class StreamEvent:
    """Event emitted during graph execution."""

    type: Literal["start", "node", "end"]
    node: Optional[str]
    ctx: Context
    step: int
    info: Dict[str, Any]
    streaming: bool = False


async def _resolve_targets(edge: EdgeTarget, ctx: Context, end: str) -> List[str]:
    """Resolve edge to list of target nodes (excluding END)."""
    if isinstance(edge, str):
        return [] if edge == end else [edge]
    if isinstance(edge, list):
        return [t for t in cast(List[str], edge) if t != end]
    result = edge(ctx)
    if isinstance(result, AwaitableABC):
        result = await result
    if isinstance(result, str):
        return [] if result == end else [result]
    return [t for t in cast(List[str], result) if t != end]


async def _invoke(node: Any, ctx: Context) -> Context:
    """Execute a node (non-streaming)."""
    if hasattr(node, "arun"):
        try:
            return await node.arun(ctx)
        except NotImplementedError:
            pass
    if hasattr(node, "astream"):
        result = ctx
        async for result in node.astream(ctx):
            pass
        return result
    if asyncio.iscoroutinefunction(node):
        return await node(ctx)
    return await asyncio.to_thread(node, ctx)


async def _invoke_streaming(
    node: Any, ctx: Context
) -> AsyncIterator[tuple[Context, bool]]:
    """Execute node, yielding (context_copy, is_final) tuples."""
    if hasattr(node, "astream"):
        result = ctx
        async for result in node.astream(ctx):
            yield result.copy(), False
        yield result.copy(), True
        return
    final = await _invoke(node, ctx)
    yield final, True


async def execute_graph(
    *,
    ctx: Context,
    nodes: Dict[str, Any],
    edges: EdgeMap,
    entry_point: str,
    max_steps: Optional[int],
    end_token: str,
    log_prefix: str,
) -> Context:
    """Execute graph and return final context."""
    final_ctx = ctx
    async for event in stream_graph_events(
        ctx=ctx,
        nodes=nodes,
        edges=edges,
        entry_point=entry_point,
        max_steps=max_steps,
        end_token=end_token,
        log_prefix=log_prefix,
    ):
        if event.type == "end":
            final_ctx = event.ctx
    return final_ctx


async def stream_graph_events(
    *,
    ctx: Context,
    nodes: Dict[str, Any],
    edges: EdgeMap,
    entry_point: str,
    max_steps: Optional[int],
    end_token: str,
    log_prefix: str,
) -> AsyncIterator[StreamEvent]:
    """Execute graph, yielding events for each node including streaming chunks."""
    execution_path: List[str] = []
    step = 0
    last_ctx = ctx.copy()
    current: List[tuple[str, Context]] = [(entry_point, ctx.copy())]

    yield StreamEvent(
        type="start",
        node=entry_point,
        ctx=ctx,
        step=0,
        info={"entry_point": entry_point},
    )

    while current:
        if max_steps is not None and step >= max_steps:
            break

        final_results: Dict[str, Context] = {}
        executable_nodes: List[tuple[str, Context]] = []

        for name, input_ctx in current:
            if name in nodes:
                executable_nodes.append((name, input_ctx))
            else:
                final_results[name] = input_ctx

        pending = len(executable_nodes)

        if pending:
            queue: asyncio.Queue[
                tuple[Optional[str], Union[Context, Exception], bool]
            ] = asyncio.Queue()

            async def run_node(name: str, input_ctx: Context) -> None:
                try:
                    async for result_ctx, is_final in _invoke_streaming(
                        nodes[name], input_ctx
                    ):
                        await queue.put((name, result_ctx, is_final))
                except Exception as exc:
                    await queue.put((None, exc, True))
                    raise

            tasks = [
                asyncio.create_task(run_node(name, input_ctx))
                for name, input_ctx in executable_nodes
            ]

            try:
                while pending > 0:
                    name, payload, is_final = await queue.get()
                    if name is None:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise payload  # type: ignore[misc]
                    if is_final:
                        final_results[name] = payload  # type: ignore[assignment]
                        pending -= 1
                        continue
                    yield StreamEvent(
                        type="node",
                        node=name,
                        ctx=payload,
                        step=step,
                        info={},
                        streaming=True,
                    )  # type: ignore[arg-type]
            finally:
                await asyncio.gather(*tasks)

        next_wave: List[tuple[str, Context]] = []

        for node_name, input_ctx in current:
            result_ctx = final_results.get(node_name, input_ctx)

            if not node_name.endswith("__END__"):
                execution_path.append(node_name)
                step += 1
                last_ctx = result_ctx
                yield StreamEvent(
                    type="node",
                    node=node_name,
                    ctx=result_ctx,
                    step=step,
                    info={},
                    streaming=False,
                )

            edge = edges.get(node_name)
            if edge is not None:
                targets = await _resolve_targets(edge, result_ctx, end_token)
                for target in targets:
                    next_wave.append((target, result_ctx.copy()))

        current = next_wave

    final_ctx = last_ctx.copy()
    final_ctx.execution_path = execution_path
    final_ctx.total_steps = len(execution_path)

    yield StreamEvent(
        type="end",
        node=None,
        ctx=final_ctx,
        step=step,
        info={"path": tuple(execution_path)},
    )


__all__ = [
    "execute_graph",
    "stream_graph_events",
    "StreamEvent",
    "EdgeTarget",
    "EdgeMap",
    "_invoke",
]
