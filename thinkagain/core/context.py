"""Context with lazy execution support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import NodeBase

from .executor import DAGExecutor
from .graph import traverse_pending


class Context:
    """Stateful execution context with lazy node execution.

    Uses PyTorch-style back-pointers for graph capture. Each Context stores
    references to its parent Contexts and the node that produced it, enabling
    proper fanout handling where shared ancestors execute only once.

    Note:
        Concurrent materialization of contexts that share ancestors is not
        supported (e.g., via ``asyncio.gather()``). Materialize sequentially.
    """

    def __init__(
        self,
        data: Context | dict | None = None,
    ):
        if isinstance(data, Context):
            # Copy from another Context - materialize first if pending
            if not data._executed:
                data._run_pending_sync()
            # Now create an isolated copy
            self._data = dict(data._data)
            self._parents: tuple[Context, ...] = ()
            self._call_args: tuple = ()
            self._call_kwargs: dict = {}
            self._node: NodeBase | None = None
            self._executed = True
            self._executor = None
        else:
            self._data = dict(data) if data is not None else {}
            self._parents = ()
            self._call_args = ()
            self._call_kwargs = {}
            self._node = None
            self._executed = True  # Root contexts with no node are executed
            self._executor = None

    def _chain(
        self,
        node: "NodeBase",
        parents: "tuple[Context, ...]",
        call_args: tuple,
        call_kwargs: dict,
    ) -> "Context":
        """Return a new Context with node and multiple parent inputs.

        Uses PyTorch-style back-pointers: the new context points to its parents
        and the node that will produce it. At execution time, we walk
        back-pointers to build the graph and execute in topological order.
        """
        ctx = Context.__new__(Context)
        ctx._data = {}  # Will be populated at execution
        ctx._parents = parents
        ctx._call_args = call_args
        ctx._call_kwargs = call_kwargs
        ctx._node = node
        ctx._executed = False
        ctx._executor = None
        return ctx

    def _get_executor(self) -> DAGExecutor:
        if self._executor is None:
            self._executor = DAGExecutor(self)
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
        """True if this context has unexecuted nodes."""
        return not self._executed

    @property
    def pending_count(self) -> int:
        """Count of unexecuted nodes in the graph leading to this context."""
        count = 0

        def increment(ctx: Context) -> None:
            nonlocal count
            count += 1

        traverse_pending(self, increment)
        return count

    @property
    def pending_names(self) -> list[str]:
        """Names of unexecuted nodes in execution order (ancestors first)."""
        names: list[str] = []
        traverse_pending(self, lambda ctx: names.append(ctx._node.name))
        return names

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
