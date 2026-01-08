"""Core error types for thinkagain."""

from __future__ import annotations


class NodeExecutionError(Exception):
    """Raised when a node fails during graph execution.

    Attributes:
        node_name: Name of the node where the error occurred
        executed: Names of nodes that completed successfully before failure
        cause: The original exception raised by the failing node
    """

    def __init__(self, node_name: str, executed: list[str], cause: Exception):
        self.node_name = node_name
        self.executed: tuple[str, ...] = tuple(executed)
        self.cause = cause
        executed_display = ", ".join(self.executed) if self.executed else "none"
        super().__init__(
            f"Node '{node_name}' failed after executing: {executed_display}.\n"
            f"Cause: {cause.__class__.__name__}: {cause}"
        )


class TracingError(Exception):
    """Raised when invalid operations are attempted during tracing.

    Examples:
    - Accessing attributes on TracedValue
    - Using Python if/while instead of cond()/while_loop() in @jit
    - Evaluating TracedValue as a boolean
    """

    pass
