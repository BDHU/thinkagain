"""
Lazy context for declarative graph construction.

LazyContext defers node execution until values are accessed,
enabling automatic graph break detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .context import Context
from .runtime import _invoke

if TYPE_CHECKING:
    from .node import Node


class NeedsMaterializationError(Exception):
    """Raised when accessing LazyContext data before materialization."""

    pass


class LazyContext:
    """
    Context that defers execution until values are accessed.

    Nodes are chained lazily via `await node(ctx)`. The chain executes
    only when you access an attribute or explicitly `await ctx`.

    Example:
        ctx = await node_a(ctx)  # pending = [a]
        ctx = await node_b(ctx)  # pending = [a, b]
        ctx = await ctx          # executes a -> b, returns materialized ctx
        print(ctx.result)        # now safe to access
    """

    def __init__(
        self,
        data: dict[str, Any],
        pending: Optional[list[Node]] = None,
    ):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_pending", pending or [])

    def _chain(self, node: Node) -> LazyContext:
        """Add node to pending chain, return new LazyContext."""
        return LazyContext(self._data.copy(), self._pending + [node])

    async def _materialize(self) -> LazyContext:
        """Execute pending chain and return materialized context."""
        if not self._pending:
            return self

        ctx = Context(**self._data)
        for node in self._pending:
            ctx = await _invoke(node.executable, ctx)

        return LazyContext(ctx.data, [])

    def __await__(self):
        """Allow `ctx = await ctx` to trigger materialization."""
        return self._materialize().__await__()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if self._pending:
            raise NeedsMaterializationError(
                f"Cannot access '{name}' with {len(self._pending)} pending node(s). "
                f"Use 'ctx = await ctx' first."
            )

        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            if self._pending:
                raise NeedsMaterializationError(
                    f"Cannot set '{name}' with pending nodes. Use 'ctx = await ctx' first."
                )
            self._data[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value, requiring materialization first."""
        if self._pending:
            raise NeedsMaterializationError(
                f"Cannot get '{key}' with pending nodes. Use 'ctx = await ctx' first."
            )
        return self._data.get(key, default)

    @property
    def data(self) -> dict[str, Any]:
        """Get data dict, requiring materialization first."""
        if self._pending:
            raise NeedsMaterializationError(
                "Cannot access data with pending nodes. Use 'ctx = await ctx' first."
            )
        return self._data.copy()

    @property
    def is_pending(self) -> bool:
        """Check if there are pending nodes to execute."""
        return bool(self._pending)

    def __repr__(self) -> str:
        if self._pending:
            names = [n.name for n in self._pending]
            return f"LazyContext(pending={names})"
        return f"LazyContext({self._data})"
