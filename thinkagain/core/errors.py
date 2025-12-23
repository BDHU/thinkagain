"""Core error types for thinkagain."""

from __future__ import annotations


class NodeDataclassError(TypeError):
    """Raised when a class-based node is not a frozen dataclass."""

    def __init__(self, node_name: str, reason: str):
        super().__init__(f"Node '{node_name}' must be a frozen dataclass. {reason}")


class NodeExecutionError(Exception):
    """Raised when a node in a pipeline fails during execution.

    Attributes:
        node_name: Name of the node where the error occurred.
        executed: Names of nodes that completed successfully before the failure.
        cause: The original exception raised by the failing node.
    """

    def __init__(self, node_name: str, executed: list[str], cause: Exception):
        self.node_name = node_name
        self.executed: tuple[str, ...] = tuple(executed)
        self.cause = cause
        executed_display = ", ".join(self.executed) if self.executed else "none"
        super().__init__(
            f"Node '{node_name}' failed after executing: {executed_display}. Cause: {cause!r}"
        )
