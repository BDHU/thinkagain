"""Graph execution for traced computation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import NodeExecutionError
from ..graph.graph import (
    Graph,
    InputRef,
    Node,
    NodeRef,
    OutputKind,
    TracedValue,
)
from ..graph.literal_refs import resolve_literal_refs
from ..traceable import map_traceable_refs


# ---------------------------------------------------------------------------
# Execution Context
# ---------------------------------------------------------------------------


@dataclass
class ExecutionContext:
    """Execution state and helpers for a single graph run."""

    graph: Graph
    inputs: tuple
    parent_values: dict[int, Any] | None = None
    node_values: dict[int, Any] = field(default_factory=dict)
    service_provider: Any | None = None  # ServiceProvider protocol
    node_ids: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.node_ids = {node.node_id for node in self.graph.nodes}

    @property
    def capture_values(self) -> dict[int, Any]:
        return (
            self.parent_values if self.parent_values is not None else self.node_values
        )

    def _resolve_input(self, index: int) -> Any:
        if index < 0 or index >= len(self.inputs):
            raise RuntimeError(f"InputRef(index={index}) out of range")
        return self.inputs[index]

    def _resolve_node_value(self, node_id: int, ref_type: str, *, strict: bool) -> Any:
        if node_id not in self.node_ids:
            raise RuntimeError(f"{ref_type}(node_id={node_id}) not in this graph")
        if strict:
            if node_id in self.node_values:
                return self.node_values[node_id]
            raise RuntimeError(f"Cannot resolve {ref_type}(node_id={node_id})")
        return self.node_values[node_id]

    def resolve(self, value: Any) -> Any:
        """Resolve a reference (TracedValue/InputRef/NodeRef) to a concrete value."""
        if isinstance(value, TracedValue):
            captured = self.graph.captured_inputs
            if captured and value.node_id in captured:
                return self._resolve_input(captured[value.node_id])
            return self._resolve_node_value(value.node_id, "TracedValue", strict=True)
        if isinstance(value, NodeRef):
            return self._resolve_node_value(value.node_id, "NodeRef", strict=False)
        if isinstance(value, InputRef):
            return self._resolve_input(value.index)
        return map_traceable_refs(value, (TracedValue, NodeRef, InputRef), self.resolve)

    def resolve_many(self, values: tuple | dict) -> tuple | dict:
        """Resolve a collection of values (args tuple or kwargs dict)."""
        if isinstance(values, dict):
            return {k: self.resolve(v) for k, v in values.items()}
        return tuple(self.resolve(v) for v in values)

    def prepare_subgraph_args(self, graph: Graph, operand_args: tuple) -> list:
        """Append captured values from the parent context to operand args."""
        return graph.append_captures(operand_args, self.capture_values)

    async def execute_node(
        self,
        node: Node,
        node_name: str,
        *,
        profiler: Any,
        profile_context: Any,
    ) -> Any:
        resolved_args = self.resolve_many(node.args)
        resolved_kwargs = self.resolve_many(node.kwargs)

        if profiler is None:
            return await node.executor.execute(resolved_args, resolved_kwargs, self)
        with profile_context(node_name):
            return await node.executor.execute(resolved_args, resolved_kwargs, self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_name(node: Node) -> str:
    """Get a display name for a node (for error messages)."""
    name = f"{node.executor.display_name()}#{node.node_id}"
    if node.source_location:
        return f"{name} at {node.source_location}"
    return name


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


async def execute_graph(
    graph: Graph,
    *args,
    parent_values: dict[int, Any] | None = None,
    service_provider: Any | None = None,
) -> Any:
    """Execute a graph with given inputs.

    Args:
        graph: Graph to execute
        *args: Input arguments
        parent_values: Values from parent graph (for nested graphs)
        service_provider: Optional service provider for service calls

    Returns:
        Graph output value
    """
    if len(args) != graph.input_count:
        raise ValueError(f"Expected {graph.input_count} inputs, got {len(args)}")
    if graph.output_ref is None:
        raise ValueError("Graph output_ref is not set.")

    ctx = ExecutionContext(
        graph=graph,
        inputs=args,
        parent_values=parent_values,
        service_provider=service_provider,
    )
    executed: list[str] = []

    # Import profiling module once if needed
    from .. import profiling

    profiler = profiling._profiler  # Direct access avoids function call overhead

    for node in graph.nodes:
        node_name = _node_name(node)

        try:
            result = await ctx.execute_node(
                node,
                node_name,
                profiler=profiler,
                profile_context=profiling.node_context,
            )
        except Exception as exc:
            raise NodeExecutionError(node_name, executed, exc) from exc

        ctx.node_values[node.node_id] = result
        executed.append(node_name)

    # Resolve output
    out = graph.output_ref
    if out.kind is OutputKind.NODE:
        return ctx.node_values[out.value]
    if out.kind is OutputKind.INPUT:
        return args[out.value]

    # LITERAL - resolve any InputRef/NodeRef in the literal
    return resolve_literal_refs(out.value, ctx.resolve)
