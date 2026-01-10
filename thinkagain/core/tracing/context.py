"""Tracing context management for graph capture."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

from ..errors import TracingError
from ..graph.graph import InputRef, Node, NodeRef, TracedValue

_trace_ctx_var: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar(
    "trace_context", default=None
)


def get_trace_context() -> "TraceContext | None":
    """Get the current trace context, or None if not tracing."""
    return _trace_ctx_var.get()


def is_tracing() -> bool:
    """Check if we're currently inside a tracing context."""
    return _trace_ctx_var.get() is not None


@dataclass
class TraceContext:
    """Context for capturing computation graph during tracing."""

    nodes: list[Node] = field(default_factory=list)
    node_counter: int = 0
    parent_ctx: "TraceContext | None" = None
    next_capture_index: int = 0
    captured_inputs: dict[int, int] = field(default_factory=dict)
    input_values: dict[TracedValue, int] = field(default_factory=dict)
    # Resource tracking (discovered during tracing) - e.g., service handles
    resources: dict[int, int] = field(
        default_factory=dict
    )  # id(resource) -> input_index
    resource_list: list = field(default_factory=list)  # Ordered list of resources

    def _normalize(self, value: Any) -> Any:
        """Convert TracedValue to appropriate reference type."""
        if not isinstance(value, TracedValue):
            return value
        if value in self.input_values:
            return InputRef(self.input_values[value])
        if value.trace_ctx is self:
            return NodeRef(value.node_id)
        if self.parent_ctx and value.trace_ctx is self.parent_ctx:
            return InputRef(self._register_capture(value))
        raise TracingError("TracedValue from unrelated trace context.")

    def _normalize_many(self, values: tuple | dict) -> tuple | dict:
        """Normalize a collection of values."""
        if isinstance(values, dict):
            return {k: self._normalize(v) for k, v in values.items()}
        return tuple(self._normalize(v) for v in values)

    def _register_capture(self, value: TracedValue) -> int:
        """Register a captured value from parent context."""
        if value.node_id not in self.captured_inputs:
            self.captured_inputs[value.node_id] = self.next_capture_index
            self.next_capture_index += 1
        return self.captured_inputs[value.node_id]

    def _next_id(self) -> int:
        """Get next node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def add_node(
        self,
        executor: Any,
        args: tuple,
        kwargs: dict,
        source_location: str | None = None,
    ) -> int:
        """Add a node to the graph with the given executor."""
        node = Node(
            node_id=self._next_id(),
            args=self._normalize_many(args),
            kwargs=self._normalize_many(kwargs),
            executor=executor,
            source_location=source_location,
        )
        self.nodes.append(node)
        return node.node_id

    def get_resource_index(self, resource: Any) -> int:
        """Get or register input index for a resource (e.g., service handle)."""
        resource_id = id(resource)

        if resource_id not in self.resources:
            resource_idx = self.next_capture_index
            self.resources[resource_id] = resource_idx
            self.resource_list.append(resource)
            self.next_capture_index += 1

        return self.resources[resource_id]
