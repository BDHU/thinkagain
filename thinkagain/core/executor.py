"""Graph execution for traced computation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import NodeExecutionError
from .graph import (
    CallNode,
    CondNode,
    Graph,
    GraphNode,
    InputRef,
    NodeRef,
    OutputKind,
    ScanNode,
    TracedValue,
    WhileNode,
)
from .runtime import maybe_await


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


def _node_name(node: GraphNode) -> str:
    """Get a display name for a node (for error messages)."""
    if isinstance(node, CallNode):
        fn_name = getattr(node.fn, "__name__", type(node.fn).__name__)
        name = f"{fn_name}#{node.node_id}"
    elif isinstance(node, CondNode):
        name = f"cond#{node.node_id}"
    elif isinstance(node, WhileNode):
        name = f"while#{node.node_id}"
    elif isinstance(node, ScanNode):
        name = f"scan#{node.node_id}"
    else:
        name = f"node#{node.node_id}"
    if node.source_location:
        return f"{name} at {node.source_location}"
    return name


# ---------------------------------------------------------------------------
# Node Executors
# ---------------------------------------------------------------------------


async def _exec_call(
    node: CallNode, args: tuple, kwargs: dict, ctx: ExecutionContext
) -> Any:
    """Execute a regular function call node."""
    return await maybe_await(node.fn, *args, **kwargs)


async def _exec_cond(
    node: CondNode, args: tuple, kwargs: dict, ctx: ExecutionContext
) -> Any:
    """Execute a conditional node."""
    operand = args[0]
    pred_result = await maybe_await(node.pred_fn, operand)
    branch = node.branches["true"] if pred_result else node.branches["false"]

    if isinstance(branch, Graph):
        graph_args = ctx.prepare_subgraph_args(branch, (operand,))
        return await branch.execute(*graph_args, parent_values=ctx.capture_values)
    return await maybe_await(branch, operand)


async def _exec_while(
    node: WhileNode, args: tuple, kwargs: dict, ctx: ExecutionContext
) -> Any:
    """Execute a while loop node."""
    operand = args[0]

    while await maybe_await(node.cond_fn, operand):
        if isinstance(node.body_fn, Graph):
            graph_args = ctx.prepare_subgraph_args(node.body_fn, (operand,))
            operand = await node.body_fn.execute(
                *graph_args, parent_values=ctx.capture_values
            )
        else:
            operand = await maybe_await(node.body_fn, operand)
    return operand


async def _exec_scan(
    node: ScanNode, args: tuple, kwargs: dict, ctx: ExecutionContext
) -> Any:
    """Execute a scan node."""
    carry, xs = args[0], args[1]
    outputs = []

    for x in xs:
        if isinstance(node.body_fn, Graph):
            graph_args = ctx.prepare_subgraph_args(node.body_fn, (carry, x))
            result = await node.body_fn.execute(
                *graph_args, parent_values=ctx.capture_values
            )
        else:
            result = await maybe_await(node.body_fn, carry, x)

        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                f"scan body must return (carry, output) tuple, got {result!r}"
            )
        carry, output = result
        outputs.append(output)

    return (carry, outputs)


# Dispatch table for node execution
_EXECUTORS = {
    CallNode: _exec_call,
    CondNode: _exec_cond,
    WhileNode: _exec_while,
    ScanNode: _exec_scan,
}


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


async def execute_graph(
    graph: Graph, *args, parent_values: dict[int, Any] | None = None
) -> Any:
    """Execute a graph with given inputs."""
    if len(args) != graph.input_count:
        raise ValueError(f"Expected {graph.input_count} inputs, got {len(args)}")
    ctx = ExecutionContext(graph=graph, inputs=args, parent_values=parent_values)
    executed: list[str] = []

    for node in graph.nodes:
        resolved_args = ctx.resolve_many(node.args)
        resolved_kwargs = ctx.resolve_many(node.kwargs)

        executor = _EXECUTORS.get(type(node))
        if not executor:
            raise ValueError(f"Unknown node type: {type(node).__name__}")

        try:
            result = await executor(node, resolved_args, resolved_kwargs, ctx)
        except Exception as exc:
            raise NodeExecutionError(_node_name(node), executed, exc) from exc

        ctx.node_values[node.node_id] = result
        executed.append(_node_name(node))

    # Resolve output
    out = graph.output_ref
    if out.kind is OutputKind.NODE:
        return ctx.node_values[out.value]
    if out.kind is OutputKind.INPUT:
        return args[out.value]
    return out.value  # LITERAL
