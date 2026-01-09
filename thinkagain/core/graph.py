"""Graph data structures for traced computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING, Protocol

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
# Node Executor Protocol
# ---------------------------------------------------------------------------


class NodeExecutor(Protocol):
    """Strategy for executing a graph node.

    This protocol defines the interface that all node executors must implement.
    Each executor encapsulates both the execution logic and the data needed
    for that execution (e.g., the function to call, branches to take, etc.).

    Executors should be immutable (frozen dataclasses) to ensure graphs are
    safely cacheable and serializable.
    """

    async def execute(
        self,
        args: tuple,
        kwargs: dict,
        ctx: Any,  # ExecutionContext, but avoids circular import
    ) -> Any:
        """Execute this node with resolved arguments.

        Args:
            args: Resolved positional arguments
            kwargs: Resolved keyword arguments
            ctx: Execution context with graph state and helpers

        Returns:
            Result of the node execution
        """
        ...

    def display_name(self) -> str:
        """Get a display name for this node (for error messages and debugging).

        Returns:
            Human-readable name for this node
        """
        ...


# ---------------------------------------------------------------------------
# Graph Node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """Universal graph node.

    A node represents a single operation in the computation graph. The actual
    execution logic is delegated to the executor, which implements the
    NodeExecutor protocol.

    Attributes:
        node_id: Unique identifier within the graph
        args: Normalized argument references (InputRef, NodeRef, or literals)
        kwargs: Normalized keyword argument references
        executor: Strategy object that knows how to execute this node
        source_location: Optional source code location for debugging
    """

    node_id: int
    args: tuple
    kwargs: dict
    executor: Any  # NodeExecutor - Any to avoid protocol complexity with dataclass
    source_location: str | None = None


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

    nodes: list[Node]
    input_count: int
    output_ref: OutputRef | None = None
    captured_inputs: dict[int, int] = field(default_factory=dict)
    resource_list: list = field(
        default_factory=list
    )  # Resources (handles, etc.) used in graph

    def __post_init__(self):
        if self.output_ref is None:
            if self.nodes:
                self.output_ref = OutputRef(OutputKind.NODE, self.nodes[-1].node_id)
            else:
                raise ValueError("Empty graph requires explicit output_ref.")

    async def execute(
        self,
        *args,
        parent_values: dict[int, Any] | None = None,
        service_provider: Any | None = None,
    ) -> Any:
        """Execute this graph with given inputs.

        Args:
            *args: Input arguments
            parent_values: Values from parent graph (for nested graphs)
            service_provider: Optional service provider for service calls

        Returns:
            Graph output value
        """
        from .executor import execute_graph

        return await execute_graph(
            self, *args, parent_values=parent_values, service_provider=service_provider
        )
