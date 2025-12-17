"""Context with lazy execution support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import Node

from .errors import NodeExecutionError, NodeSignatureError


class Context:
    """Stateful execution context with lazy node execution."""

    __slots__ = ("_data", "_pending")

    def __init__(self, data: Context | dict | None = None):
        if isinstance(data, Context):
            # Copy from another Context (always copies data for isolation)
            object.__setattr__(self, "_data", dict(data._data))
            object.__setattr__(self, "_pending", list(data._pending))
        else:
            object.__setattr__(self, "_data", data if data is not None else {})
            object.__setattr__(self, "_pending", [])

    def _chain(self, node: "Node") -> "Context":
        """Return a new Context that shares data but has this node pending."""
        ctx = Context(self._data)  # shares underlying data
        object.__setattr__(ctx, "_pending", self._pending + [node])
        return ctx

    def copy(self) -> "Context":
        """Create a copy with isolated data (no shared mutations)."""
        return Context(self)

    async def _run_pending_async(self) -> None:
        if not self._pending:
            return

        executed: list[str] = []
        ctx = Context(self._data)

        for node in self._pending:
            try:
                ctx = await node.execute(ctx)
                executed.append(node.name)
                # Check if node returned ctx with unawaited pending nodes
                if ctx._pending:
                    pending_names = [n.name for n in ctx._pending]
                    raise RuntimeError(
                        f"Node '{node.name}' returned context with unawaited pending nodes: {pending_names}. "
                        f"Use 'await ctx' before returning from a node that calls other nodes."
                    )
            except (NodeExecutionError, NodeSignatureError):
                raise
            except Exception as e:
                raise NodeExecutionError(node.name, executed, e) from e

        object.__setattr__(self, "_data", ctx._data)
        object.__setattr__(self, "_pending", [])

    def _run_pending_sync(self) -> None:
        if not self._pending:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._run_pending_async())
        else:
            raise RuntimeError(
                "Cannot materialize synchronously from async context. "
                "Use 'await ctx' or 'await arun(pipeline)' instead."
            )

    def materialize(self) -> "Context":
        """Synchronously execute all pending nodes and return this context."""
        self._run_pending_sync()
        return self

    async def amaterialize(self) -> "Context":
        """Asynchronously execute all pending nodes and return this context."""
        await self._run_pending_async()
        return self

    def __await__(self):
        """Awaiting a Context is equivalent to calling `amaterialize`."""
        return self.amaterialize().__await__()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context, materializing pending nodes first."""
        self._run_pending_sync()
        return self._data.get(key, default)

    async def aget(self, key: str, default: Any = None) -> Any:
        """Async get - materializes pending nodes first."""
        await self._run_pending_async()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value on the context, materializing pending nodes first."""
        self._run_pending_sync()
        self._data[key] = value

    async def aset(self, key: str, value: Any) -> None:
        """Async set - materializes pending nodes first."""
        await self._run_pending_async()
        self._data[key] = value

    def peek(self, key: str, default: Any = None) -> Any:
        """Get value without materializing pending nodes."""
        return self._data.get(key, default)

    @property
    def is_pending(self) -> bool:
        return bool(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_names(self) -> list[str]:
        return [n.name for n in self._pending]

    def keys(self) -> list[str]:
        self._run_pending_sync()
        return list(self._data.keys())

    def values(self) -> list[Any]:
        self._run_pending_sync()
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        self._run_pending_sync()
        return list(self._data.items())

    def to_dict(self) -> dict:
        """Return a shallow copy of the underlying data."""
        self._run_pending_sync()
        return dict(self._data)

    def __iter__(self):
        self._run_pending_sync()
        return iter(self._data)

    def __repr__(self) -> str:
        self._run_pending_sync()
        return f"Context({self._data})"

    def __len__(self) -> int:
        self._run_pending_sync()
        return len(self._data)
