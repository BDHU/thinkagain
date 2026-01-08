"""Graph data structures for traced computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .tracing import TraceContext


class OutputKind(Enum):
    """Kind of output reference in a graph."""

    NODE = "node"
    INPUT = "input"
    LITERAL = "literal"


@dataclass(frozen=True)
class InputRef:
    """Reference to a positional input captured during tracing."""

    index: int


@dataclass(frozen=True)
class NodeRef:
    """Reference to a node output in a graph."""

    node_id: int


@dataclass(frozen=True)
class OutputRef:
    """Reference to the output of a graph."""

    kind: OutputKind
    value: Any


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """Base class for traced computation graph nodes."""

    node_id: int
    args: tuple
    kwargs: dict
    source_location: str | None = None


@dataclass
class CallNode(GraphNode):
    """Regular function call node."""

    fn: Callable = field(default=None)


@dataclass
class CondNode(GraphNode):
    """Conditional control flow node."""

    pred_fn: Callable = field(default=None)
    branches: dict[str, "Graph | Callable"] = field(default_factory=dict)


@dataclass
class WhileNode(GraphNode):
    """While loop control flow node."""

    cond_fn: Callable = field(default=None)
    body_fn: "Graph | Callable" = field(default=None)


@dataclass
class ScanNode(GraphNode):
    """Scan control flow node."""

    body_fn: "Graph | Callable" = field(default=None)


# ---------------------------------------------------------------------------
# TracedValue
# ---------------------------------------------------------------------------


class TracedValue:
    """Placeholder for values during tracing.

    Represents a value that will be computed when the graph executes.
    Operations on TracedValue during tracing raise errors with helpful messages.
    """

    __slots__ = ("node_id", "trace_ctx")

    def __init__(self, node_id: int, trace_ctx: "TraceContext"):
        self.node_id = node_id
        self.trace_ctx = trace_ctx

    def __repr__(self) -> str:
        return f"TracedValue(node_id={self.node_id})"

    def _error(self, msg: str) -> None:
        from .errors import TracingError

        raise TracingError(msg)

    def __getattr__(self, name: str):
        self._error(
            f"Cannot access attribute '{name}' on TracedValue during tracing. "
            "Use lambda functions for attribute access in control flow."
        )

    def __bool__(self):
        self._error(
            "Cannot evaluate TracedValue as boolean during tracing. "
            "Use cond() instead of Python if/else inside @jit functions."
        )

    def __await__(self):
        self._error(
            "Cannot await TracedValue during tracing. Await the underlying node."
        )
        yield

    def __getitem__(self, key):
        self._error(
            f"Cannot index TracedValue during tracing. Use lambda: lambda s: s[{key!r}]"
        )

    def __len__(self):
        self._error("Cannot get length of TracedValue during tracing.")

    def __iter__(self):
        self._error(
            "Cannot iterate over TracedValue during tracing. Use scan() for loops."
        )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class Graph:
    """A compiled computation graph.

    Represents both top-level graphs (from @jit) and subgraphs (from control flow).
    """

    nodes: list[GraphNode]
    input_count: int
    output_ref: OutputRef | None = None
    captured_inputs: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.output_ref is None:
            if self.nodes:
                self.output_ref = OutputRef(OutputKind.NODE, self.nodes[-1].node_id)
            else:
                raise ValueError("Empty graph requires explicit output_ref.")

    async def execute(self, *args, parent_values: dict[int, Any] | None = None) -> Any:
        """Execute this graph with given inputs."""
        from .executor import execute_graph

        return await execute_graph(self, *args, parent_values=parent_values)
