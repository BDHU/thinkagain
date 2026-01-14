"""Execution context for pools, hooks, and profiling state."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionContext:
    """Context-local state for execution infrastructure."""

    pools: dict[tuple[object, ...], Any] = field(default_factory=dict)
    hooks: list[Any] = field(default_factory=list)
    profiler: Any | None = None


_execution_context: contextvars.ContextVar[ExecutionContext | None] = (
    contextvars.ContextVar("execution_context", default=None)
)


def get_current_execution_context() -> ExecutionContext:
    """Get or create the current execution context."""
    ctx = _execution_context.get()
    if ctx is None:
        ctx = ExecutionContext()
        _execution_context.set(ctx)
    return ctx


def set_current_execution_context(
    ctx: ExecutionContext | None,
) -> contextvars.Token:
    """Set the execution context for the current scope."""
    return _execution_context.set(ctx)


def reset_current_execution_context(token: contextvars.Token) -> None:
    """Reset the execution context using a stored token."""
    _execution_context.reset(token)
