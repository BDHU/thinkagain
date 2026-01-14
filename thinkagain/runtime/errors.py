"""Core error types for thinkagain."""

from __future__ import annotations


class ThinkAgainError(Exception):
    """Base exception for all thinkagain errors."""

    pass


class OpExecutionError(ThinkAgainError):
    """Raised when an operation fails during execution."""

    def __init__(self, op_name: str, executed: list[str], cause: Exception):
        self.op_name = op_name
        self.executed: tuple[str, ...] = tuple(executed)
        self.cause = cause
        executed_display = ", ".join(self.executed) if self.executed else "none"
        super().__init__(
            f"Op '{op_name}' failed after executing: {executed_display}.\n"
            f"Cause: {cause.__class__.__name__}: {cause}"
        )


class SchedulerError(ThinkAgainError):
    """Raised when scheduler encounters an error."""

    pass


class PoolError(ThinkAgainError):
    """Raised when service pool encounters an error."""

    pass


class MeshError(ThinkAgainError):
    """Raised when mesh configuration is invalid."""

    pass


class ResourceError(ThinkAgainError):
    """Raised when resources are unavailable."""

    pass
