"""Context with lazy execution support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import Node


class Context:
    """Dict-like context with auto-materializing lazy execution."""

    __slots__ = ("_data", "_pending")

    def __init__(self, data: dict | None = None, pending: list[Node] | None = None):
        object.__setattr__(self, "_data", dict(data) if data else {})
        object.__setattr__(self, "_pending", pending or [])

    def _chain(self, node: Node) -> Context:
        return Context(self._data.copy(), self._pending + [node])

    async def _async_sync(self) -> None:
        """Materialize pending nodes asynchronously."""
        if not self._pending:
            return
        ctx = Context(self._data)
        for node in self._pending:
            ctx = await node.execute(ctx)
        object.__setattr__(self, "_data", ctx._data)
        object.__setattr__(self, "_pending", [])

    def _sync(self) -> None:
        """Materialize pending nodes synchronously."""
        if not self._pending:
            return

        async def _run():
            ctx = Context(self._data)
            for node in self._pending:
                ctx = await node.execute(ctx)
            return ctx

        # Run in event loop
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run()).result()
        except RuntimeError:
            result = asyncio.run(_run())

        object.__setattr__(self, "_data", result._data)
        object.__setattr__(self, "_pending", [])

    def __getattr__(self, name: str) -> Any:
        self._sync()
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._sync()
        self._data[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        self._sync()
        return self._data.get(key, default)

    @property
    def is_pending(self) -> bool:
        return bool(self._pending)
