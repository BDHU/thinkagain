"""Lazy context for deferred execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .context import Context

if TYPE_CHECKING:
    from .node import Node


class NeedsMaterializationError(Exception):
    """Raised when accessing data before materialization."""
    pass


class LazyContext:
    """Context that defers execution until awaited."""

    __slots__ = ("_data", "_pending")

    def __init__(self, data: dict, pending: list[Node] | None = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_pending", pending or [])

    def _chain(self, node: Node) -> LazyContext:
        return LazyContext(self._data.copy(), self._pending + [node])

    async def _materialize(self) -> LazyContext:
        if not self._pending:
            return self
        ctx = Context(self._data)
        for node in self._pending:
            ctx = await node.fn(ctx)
        return LazyContext(dict(ctx), [])

    def __await__(self):
        return self._materialize().__await__()

    def __getattr__(self, name: str) -> Any:
        if self._pending:
            raise NeedsMaterializationError(
                f"Cannot access '{name}' - await ctx first"
            )
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if self._pending:
            raise NeedsMaterializationError("Cannot set - await ctx first")
        self._data[name] = value

    @property
    def data(self) -> dict:
        if self._pending:
            raise NeedsMaterializationError("Cannot access data - await ctx first")
        return self._data.copy()

    @property
    def is_pending(self) -> bool:
        return bool(self._pending)
