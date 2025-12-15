"""
Node wrapper for declarative graph construction.

Nodes wrap executables and enable lazy chaining.
"""

from __future__ import annotations

from typing import Any, Union

from .lazy import LazyContext


class Node:
    """
    Declarative node wrapper for executables.

    Wraps any executable (Executable subclass, async function, etc.)
    and enables lazy chaining when called with a LazyContext.

    Example:
        retrieve = Node(RetrieveDocs())
        generate = Node(GenerateAnswer())

        async def pipeline(ctx):
            ctx = await retrieve(ctx)  # lazy - adds to pending
            ctx = await generate(ctx)  # lazy - adds to pending
            return ctx                 # still pending until materialized
    """

    def __init__(self, executable: Any, name: str | None = None):
        self.executable = executable
        self.name = name or self._infer_name(executable)

    async def __call__(self, ctx: Union[LazyContext, dict]) -> LazyContext:
        """
        Add this node to the pending chain.

        Does not execute - just records that this node should run.
        """
        if isinstance(ctx, dict):
            ctx = LazyContext(ctx, [])

        return ctx._chain(self)

    @staticmethod
    def _infer_name(executable: Any) -> str:
        """Infer a name from the executable."""
        if hasattr(executable, "name"):
            return executable.name
        if hasattr(executable, "__name__"):
            return executable.__name__
        return executable.__class__.__name__

    def __repr__(self) -> str:
        return f"Node({self.name!r})"
