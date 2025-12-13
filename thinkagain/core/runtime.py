"""
Runtime helpers shared by Graph and CompiledGraph.

This module centralizes the mechanics for executing graph structures so
that builder-oriented classes (Graph) and immutable executors
(CompiledGraph) can share the exact same behavior. Keeping the execution
loop, logging, and utility helpers in one place reduces duplication and
lowers the surface area for bugs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional, Union

from .context import Context


@dataclass
class DirectEdge:
    """Direct edge to a single target node."""

    target: str


@dataclass
class ConditionalEdge:
    """Conditional edge that routes based on context state."""

    route_fn: Callable[[Context], str]
    paths: Dict[str, str]


EdgeTarget = Union[
    DirectEdge, ConditionalEdge, str
]  # str for backward compat during migration


@dataclass
class StreamEvent:
    """Structured event emitted while a graph executes."""

    type: Literal["start", "node", "end"]
    node: Optional[str]
    ctx: Context
    step: int
    info: Dict[str, Any]
    streaming: bool = False  # True if this is an intermediate streaming update


async def execute_graph(
    *,
    ctx: Context,
    nodes: Dict[str, Any],
    edges: Dict[str, EdgeTarget],
    entry_point: Optional[str],
    max_steps: Optional[int],
    end_token: str,
    log_prefix: str,
) -> Context:
    """
    Execute a graph or compiled graph with shared semantics.

    Args:
        ctx: Context instance to mutate while running.
        nodes: Mapping of node names to executables/callables.
        edges: Mapping of node names to next node (direct or conditional).
        entry_point: Node to start from.
        max_steps: Optional guard against infinite loops.
        end_token: Sentinel string representing graph termination.
        log_prefix: Text inserted before log messages (e.g. "[Graph:rag]").
    """
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
        final_ctx = event.ctx

    return final_ctx


async def stream_graph_events(
    *,
    ctx: Context,
    nodes: Dict[str, Any],
    edges: Dict[str, EdgeTarget],
    entry_point: Optional[str],
    max_steps: Optional[int],
    end_token: str,
    log_prefix: str,
) -> AsyncIterator[StreamEvent]:
    """Yield structured events as the graph executes."""
    if entry_point is None:
        raise ValueError("Entry point not set. Use set_entry() before execution.")

    def _log(message: str) -> None:
        ctx.log(f"{log_prefix} {message}")

    current = entry_point
    execution_path: list[str] = []

    _log("Starting execution")
    _log(f"Entry point: {current}")
    yield StreamEvent(
        type="start",
        node=current,
        ctx=ctx,
        step=0,
        info={"entry_point": entry_point},
    )

    step = 0
    while True:
        if current in (None, end_token):
            _log(f"Reached END after {step} steps")
            break

        # Check if this is a virtual END node (from flattened subgraphs)
        # These nodes exist only as edge targets, not as executable nodes
        if current not in nodes and current.endswith("__END__"):
            _log(f"Passing through virtual end node: {current}")
            # Don't execute anything, just follow the edge
            next_node = await _resolve_next_node(edges, current, ctx, end_token, _log)
            if next_node in (None, end_token):
                break
            current = next_node
            continue

        # Execute node with streaming support
        final_ctx = ctx
        async for ctx_snapshot in _execute_node_stream(nodes, current, ctx, _log):
            final_ctx = ctx_snapshot
            # Check if this is the last yield (final result)
            # We'll yield intermediate updates with streaming=True
            # For now, yield all of them and let the consumer decide
            yield StreamEvent(
                type="node",
                node=current,
                ctx=ctx_snapshot,
                step=len(execution_path) + 1,
                info={"log_prefix": log_prefix},
                streaming=True,  # Intermediate update
            )

        # Update context to final result
        ctx = final_ctx
        execution_path.append(current)

        # Yield final node completion event
        yield StreamEvent(
            type="node",
            node=current,
            ctx=ctx,
            step=len(execution_path),
            info={"log_prefix": log_prefix},
            streaming=False,  # Final completion
        )

        next_node = await _resolve_next_node(edges, current, ctx, end_token, _log)
        if next_node in (None, end_token):
            break

        current = next_node
        step += 1

        if max_steps is not None and step >= max_steps:
            _log(f"WARNING: Terminated after max_steps={max_steps}")
            _log("This may indicate an infinite loop")
            break

    ctx.execution_path = execution_path
    ctx.total_steps = len(execution_path)

    _log("Completed execution")
    _log(f"Total steps: {ctx.total_steps}")
    path_display = " → ".join(execution_path) or "(none)"
    _log(f"Path: {path_display}")

    yield StreamEvent(
        type="end",
        node=None,
        ctx=ctx,
        step=ctx.total_steps,
        info={"path": tuple(execution_path)},
    )


async def _execute_node_stream(
    nodes: Dict[str, Any],
    node_name: str,
    ctx: Context,
    log: Callable[[str], None],
) -> AsyncIterator[Context]:
    """Execute a node and yield context snapshots during execution."""
    node = nodes[node_name]

    # Delay Graph import to avoid cycles.
    from .graph import Graph
    from .compiled_graph import CompiledGraph

    if isinstance(node, Graph):
        log(f"Entering subgraph: {node_name} ({node.name})")
    elif isinstance(node, CompiledGraph):
        log(f"Entering compiled subgraph: {node_name} ({node.name})")
    else:
        log(f"Executing: {node_name}")

    try:
        # Check if node supports streaming (has astream method)
        if hasattr(node, "astream"):
            async for ctx_snapshot in node.astream(ctx):
                yield ctx_snapshot
        else:
            # Fallback to regular invocation for non-Worker nodes
            result = await _invoke(node, ctx)
            yield result
    except Exception as exc:  # pragma: no cover - logs and re-raises exceptions
        log(f"Error in node '{node_name}' ({type(exc).__name__}): {exc}")
        raise


async def _resolve_next_node(
    edges: Dict[str, EdgeTarget],
    current: str,
    ctx: Context,
    end_token: str,
    log: Callable[[str], None],
) -> Optional[str]:
    edge = edges.get(current)

    if edge is None:
        log(f"Node '{current}' has no outgoing edge, terminating")
        return None

    # Handle ConditionalEdge
    if isinstance(edge, ConditionalEdge):
        try:
            route_result = await _call_route(edge.route_fn, ctx)
        except Exception as exc:
            log(f"Error in routing function: {exc}")
            raise

        log(f"Conditional route from '{current}': '{route_result}'")

        if route_result == end_token:
            return end_token

        if route_result in edge.paths:
            return edge.paths[route_result]

        available = list(edge.paths.keys()) + [end_token]
        raise ValueError(
            f"Route function returned '{route_result}' but no matching edge. "
            f"Available paths: {available}"
        )

    # Handle DirectEdge
    if isinstance(edge, DirectEdge):
        log(f"Direct edge: '{current}' → '{edge.target}'")
        return edge.target

    # Backward compat: plain string
    log(f"Direct edge: '{current}' → '{edge}'")
    return edge


async def _invoke(node: Any, ctx: Context) -> Context:
    if hasattr(node, "arun"):
        return await node.arun(ctx)
    if asyncio.iscoroutinefunction(node):
        return await node(ctx)
    # Sync fallback: run in thread pool
    return await asyncio.to_thread(node, ctx)


async def _call_route(route: Callable[[Context], str], ctx: Context) -> str:
    if asyncio.iscoroutinefunction(route):
        return await route(ctx)
    return route(ctx)


__all__ = [
    "execute_graph",
    "EdgeTarget",
    "DirectEdge",
    "ConditionalEdge",
    "StreamEvent",
    "stream_graph_events",
    "_invoke",
]
