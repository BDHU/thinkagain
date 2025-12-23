"""Graph traversal utilities for Context DAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .context import Context


def traverse_pending(
    ctx: "Context",
    visitor: Callable[["Context"], None] | None = None,
) -> list["Context"]:
    """Walk back-pointers and return pending contexts in topological order.

    Visits each context exactly once, parents before children, and returns only
    unexecuted contexts with nodes. Optionally calls visitor(ctx) for each
    pending context with a node.

    Args:
        ctx: The target context to traverse from.
        visitor: Optional callback invoked for each pending context that has a node.

    Returns:
        List of pending contexts in topological order (parents first).
    """
    contexts: list[Context] = []
    visited: set[int] = set()

    def collect(c: "Context") -> None:
        ctx_id = id(c)
        if ctx_id in visited:
            return
        visited.add(ctx_id)

        for parent in c._parents:
            collect(parent)

        if c._node is not None and not c._executed:
            contexts.append(c)
            if visitor is not None:
                visitor(c)

    collect(ctx)
    return contexts
