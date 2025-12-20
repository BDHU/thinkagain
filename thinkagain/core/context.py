"""Context with lazy execution support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import NodeBase

from .executor import PendingExecutor
from .metadata import ExecutionMetadata


class Context:
    """Stateful execution context with lazy node execution."""

    def __init__(
        self,
        data: Context | dict | None = None,
        *,
        metadata: ExecutionMetadata | None = None,
    ):
        if isinstance(data, Context):
            # Copy from another Context (always copies data for isolation)
            self._data = dict(data._data)
            self._pending = list(data._pending)
            self._metadata = data._metadata.copy()
            self._executor = None
        else:
            self._data = data if data is not None else {}
            self._pending = []
            self._metadata = metadata if metadata is not None else ExecutionMetadata()
            self._executor = None

    def _chain(self, node: "NodeBase") -> "Context":
        """Return a new Context that shares data/metadata but has this node pending.

        Metadata is shared (not copied) during chaining to avoid overhead.
        Copy happens only at execution time in _run_pending_async.
        """
        ctx = Context.__new__(Context)
        ctx._data = self._data
        ctx._metadata = self._metadata  # shared, not copied
        ctx._pending = self._pending + [node]
        ctx._executor = None
        return ctx

    def _get_executor(self) -> PendingExecutor:
        if self._executor is None:
            self._executor = PendingExecutor(self)
        return self._executor

    def copy(self) -> "Context":
        """Create a copy with isolated data (no shared mutations)."""
        return Context(self)

    async def _run_pending_async(self) -> None:
        await self._get_executor().run_async()

    def _run_pending_sync(self) -> None:
        self._get_executor().run_sync()

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

    def delete(self, key: str) -> None:
        """Delete a key from the context, materializing pending nodes first."""
        self._run_pending_sync()
        self._data.pop(key, None)

    async def adelete(self, key: str) -> None:
        """Async delete - materializes pending nodes first."""
        await self._run_pending_async()
        self._data.pop(key, None)

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

    @property
    def metadata(self) -> ExecutionMetadata:
        """Return the execution metadata."""
        return self._metadata
