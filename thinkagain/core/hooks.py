"""Execution hooks for pluggable extensions.

Provides a lightweight hook system for extending executor behavior without
tight coupling. Extensions (like distributed execution) can register hooks
to intercept and modify execution flow.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

__all__ = ["ExecutionHook", "register_hook", "unregister_hook", "get_hooks"]


class ExecutionHook(Protocol):
    """Protocol for execution hooks.

    Hooks can intercept function execution and either:
    1. Return (True, result) to short-circuit execution with a custom result
    2. Return (False, None) to continue with normal execution
    """

    async def __call__(
        self,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        node_id: int | None = None,
    ) -> tuple[bool, Any]:
        """Intercept function execution.

        Args:
            fn: Function being executed
            args: Positional arguments
            kwargs: Keyword arguments
            node_id: Optional node ID for context

        Returns:
            (handled, result) tuple:
            - If handled=True, result is used instead of calling fn
            - If handled=False, normal execution continues
        """
        ...


# Global hook registry (order matters - first match wins)
_hooks: list[ExecutionHook] = []


def register_hook(hook: ExecutionHook) -> None:
    """Register an execution hook.

    Hooks are checked in registration order. The first hook that returns
    (True, result) will short-circuit execution.

    Args:
        hook: Hook to register
    """
    if hook not in _hooks:
        _hooks.append(hook)


def unregister_hook(hook: ExecutionHook) -> None:
    """Unregister an execution hook.

    Args:
        hook: Hook to remove
    """
    if hook in _hooks:
        _hooks.remove(hook)


def get_hooks() -> list[ExecutionHook]:
    """Get all registered hooks (for testing/debugging)."""
    return _hooks.copy()


def clear_hooks() -> None:
    """Clear all hooks (for testing)."""
    _hooks.clear()
