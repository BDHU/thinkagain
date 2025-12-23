"""Graph traversal utilities for Context DAGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


def traverse_pending(ctx: "Context") -> list["Context"]:
    """Walk back-pointers and return pending contexts in topological order.

    Visits each context exactly once, parents before children, and returns only
    unexecuted contexts with nodes.
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

    collect(ctx)
    return contexts
