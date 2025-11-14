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
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .context import Context
from .executable import run_sync

EdgeMap = Dict[str, str]
ConditionalEdge = Tuple[Callable[[Context], str], EdgeMap]
EdgeTarget = Union[str, ConditionalEdge]


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
    if entry_point is None:
        raise ValueError("Entry point not set. Use set_entry() before execution.")

    def _log(message: str) -> None:
        ctx.log(f"{log_prefix} {message}")

    current = entry_point
    execution_path: list[str] = []

    _log("Starting execution")
    _log(f"Entry point: {current}")

    step = 0
    while True:
        if current in (None, end_token):
            _log(f"Reached END after {step} steps")
            break

        ctx = await _execute_node(nodes, current, ctx, _log)
        execution_path.append(current)

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
    return ctx


async def _execute_node(
    nodes: Dict[str, Any],
    node_name: str,
    ctx: Context,
    log: Callable[[str], None],
) -> Context:
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
        return await _invoke(node, ctx)
    except Exception as exc:  # pragma: no cover - passthrough
        log(f"Error in node '{node_name}': {exc}")
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

    if isinstance(edge, tuple):
        route_fn, edge_map = edge
        try:
            route_result = await _call_route(route_fn, ctx)
        except Exception as exc:
            log(f"Error in routing function: {exc}")
            raise

        log(f"Conditional route from '{current}': '{route_result}'")

        if route_result in edge_map:
            return edge_map[route_result]
        if route_result == end_token:
            return end_token

        available = list(edge_map.keys()) + [end_token]
        raise ValueError(
            f"Route function returned '{route_result}' but no matching edge. "
            f"Available paths: {available}"
        )

    log(f"Direct edge: '{current}' → '{edge}'")
    return edge


async def _invoke(node: Any, ctx: Context) -> Context:
    if hasattr(node, "arun"):
        return await node.arun(ctx)
    if asyncio.iscoroutinefunction(node):
        return await node(ctx)
    return await run_sync(node, ctx)


async def _call_route(route: Callable[[Context], str], ctx: Context) -> str:
    if asyncio.iscoroutinefunction(route):
        return await route(ctx)
    return route(ctx)


__all__ = ["execute_graph", "EdgeTarget"]
