"""Graph execution for traced computation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import NodeExecutionError
from .graph import (
    Graph,
    InputRef,
    Node,
    NodeRef,
    OutputKind,
    TracedValue,
)


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

    def resolve(self, value: Any) -> Any:
        """Resolve a reference (TracedValue/InputRef/NodeRef) to a concrete value."""
        if isinstance(value, TracedValue):
            captured = self.graph.captured_inputs
            if captured and value.node_id in captured:
                return self.inputs[captured[value.node_id]]
            if value.node_id not in self.node_ids:
                raise RuntimeError(
                    f"TracedValue(node_id={value.node_id}) not in this graph"
                )
            if value.node_id in self.node_values:
                return self.node_values[value.node_id]
            raise RuntimeError(f"Cannot resolve TracedValue(node_id={value.node_id})")
        if isinstance(value, NodeRef):
            if value.node_id not in self.node_ids:
                raise RuntimeError(
                    f"NodeRef(node_id={value.node_id}) not in this graph"
                )
            return self.node_values[value.node_id]
        if isinstance(value, InputRef):
            if value.index < 0 or value.index >= len(self.inputs):
                raise RuntimeError(f"InputRef(index={value.index}) out of range")
            return self.inputs[value.index]
        return value

    def resolve_many(self, values: tuple | dict) -> tuple | dict:
        """Resolve a collection of values (args tuple or kwargs dict)."""
        if isinstance(values, dict):
            return {k: self.resolve(v) for k, v in values.items()}
        return tuple(self.resolve(v) for v in values)

    def prepare_subgraph_args(self, graph: Graph, operand_args: tuple) -> list:
        """Append captured values from the parent context to operand args."""
        args = list(operand_args)
        capture_values = self.capture_values
        for parent_id in sorted(graph.captured_inputs, key=graph.captured_inputs.get):
            args.append(capture_values[parent_id])
        return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_literal_output(value: Any, ctx: ExecutionContext) -> Any:
    """Recursively resolve refs (InputRef/NodeRef) in a literal output value.

    This handles cases where a @jit function returns a literal container
    (dict, list, tuple) that contains InputRef or NodeRef objects as elements.
    These need to be resolved to their actual computed values.
    """
    if isinstance(value, (InputRef, NodeRef)):
        # Resolve ref to its computed value
        return ctx.resolve(value)
    elif isinstance(value, dict):
        # Recursively resolve dict values
        return {k: _resolve_literal_output(v, ctx) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively resolve list elements
        return [_resolve_literal_output(v, ctx) for v in value]
    elif isinstance(value, tuple):
        # Recursively resolve tuple elements
        return tuple(_resolve_literal_output(v, ctx) for v in value)
    else:
        # Primitive value, return as-is
        return value


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

    ctx = ExecutionContext(
        graph=graph,
        inputs=args,
        parent_values=parent_values,
        service_provider=service_provider,
    )
    executed: list[str] = []

    # Import profiling module once if needed
    from . import profiling

    profiler = profiling._profiler  # Direct access avoids function call overhead

    for node in graph.nodes:
        resolved_args = ctx.resolve_many(node.args)
        resolved_kwargs = ctx.resolve_many(node.kwargs)

        try:
            # Fast path: no profiling overhead when disabled
            if profiler is None:
                result = await node.executor.execute(
                    resolved_args, resolved_kwargs, ctx
                )
            else:
                # Profiling enabled - use context manager
                node_name = _node_name(node)
                with profiling.node_context(node_name):
                    result = await node.executor.execute(
                        resolved_args, resolved_kwargs, ctx
                    )
        except Exception as exc:
            node_name = _node_name(node)
            raise NodeExecutionError(node_name, executed, exc) from exc

        ctx.node_values[node.node_id] = result
        if profiler is not None:
            executed.append(node_name if "node_name" in locals() else _node_name(node))
        else:
            executed.append(_node_name(node))

    # Resolve output
    out = graph.output_ref
    if out.kind is OutputKind.NODE:
        return ctx.node_values[out.value]
    if out.kind is OutputKind.INPUT:
        return args[out.value]

    # LITERAL - need to resolve any TracedValues in the literal
    return _resolve_literal_output(out.value, ctx)
