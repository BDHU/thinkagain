"""
Runtime execution engine for compiled graphs.

Executes nodes, following edges until reaching END.
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
)

from .context import Context

# Edge target: static string(s) or dynamic callable
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
    """Resolve edge to list of target nodes."""
    if isinstance(edge, str):
        return [] if edge == end else [edge]
    elif isinstance(edge, list):
        return [t for t in edge if t != end]
    else:
        result = edge(ctx)
        if isinstance(result, AwaitableABC):
            result = await result
        if isinstance(result, str):
            return [] if result == end else [result]
        return [t for t in result if t != end]


async def _invoke(node: Any, ctx: Context) -> Context:
    """Execute a node."""
    if hasattr(node, "arun"):
        try:
            return await node.arun(ctx)
        except NotImplementedError:
            pass
    # Fallback to astream (consume all, return last)
    if hasattr(node, "astream"):
        result = ctx
        async for result in node.astream(ctx):
            pass
        return result
    if asyncio.iscoroutinefunction(node):
        return await node(ctx)
    return await asyncio.to_thread(node, ctx)


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
        ctx=ctx, nodes=nodes, edges=edges, entry_point=entry_point,
        max_steps=max_steps, end_token=end_token, log_prefix=log_prefix,
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
    """Execute graph, yielding events for each node."""

    execution_path: List[str] = []
    step = 0
    last_ctx = ctx.copy()

    # Current wave of execution: list of (node_name, input_context)
    current: List[tuple[str, Context]] = [(entry_point, ctx.copy())]

    yield StreamEvent(
        type="start", node=entry_point, ctx=ctx, step=0,
        info={"entry_point": entry_point},
    )

    while current:
        if max_steps is not None and step >= max_steps:
            break

        # Execute all nodes in current wave in parallel
        async def run_node(name: str, input_ctx: Context) -> tuple[str, Context]:
            # Handle virtual END nodes from flattening
            if name not in nodes and name.endswith("__END__"):
                return name, input_ctx
            return name, await _invoke(nodes[name], input_ctx)

        results = await asyncio.gather(*[
            run_node(name, input_ctx) for name, input_ctx in current
        ])

        # Collect next wave
        next_wave: List[tuple[str, Context]] = []

        for node_name, result_ctx in results:
            if not node_name.endswith("__END__"):
                execution_path.append(node_name)
                step += 1
                last_ctx = result_ctx

                yield StreamEvent(
                    type="node", node=node_name, ctx=result_ctx, step=step,
                    info={}, streaming=False,
                )

            # Get successors
            edge = edges.get(node_name)
            if edge is not None:
                targets = await _resolve_targets(edge, result_ctx, end_token)
                for target in targets:
                    next_wave.append((target, result_ctx.copy()))

        current = next_wave

    # Use the last executed context as final
    final_ctx = last_ctx.copy()
    final_ctx.execution_path = execution_path
    final_ctx.total_steps = len(execution_path)

    yield StreamEvent(
        type="end", node=None, ctx=final_ctx, step=step,
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
