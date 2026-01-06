"""Context with lazy execution support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ContextManager, Generic, TypeVar

if TYPE_CHECKING:
    from .node import NodeBase

from .executor import DAGExecutor

T = TypeVar("T")


class Context(Generic[T]):
    """Lazy wrapper around any value.

    Context holds a single value of any type and tracks the computation graph
    via PyTorch-style back-pointers. The value is computed lazily when accessed.

    Example:
        ctx = Context(5)
        ctx = add_one(ctx)  # Lazy - not computed yet
        ctx = double(ctx)   # Still lazy
        print(ctx.data)     # Materializes: 12

    For structured data, use any Python type:
        ctx = Context({"query": "...", "docs": []})
        ctx = Context(MyDataclass(field1=..., field2=...))
    """

    __slots__ = (
        "_data",
        "_parents",
        "_call_args",
        "_call_kwargs",
        "_node",
        "_executed",
        "_executor",
        "_context_factory",
    )

    def __init__(
        self,
        data: Any = None,
        *,
        context_factory: Callable[[str], ContextManager[None]] | None = None,
    ):
        self._data = data
        self._parents: tuple[Context, ...] = ()
        self._call_args: tuple = ()
        self._call_kwargs: dict = {}
        self._node: NodeBase | None = None
        self._executed = True  # Root contexts are already "executed"
        self._executor: DAGExecutor | None = None
        self._context_factory = context_factory

    def _chain(
        self,
        node: "NodeBase",
        parents: "tuple[Context, ...]",
        call_args: tuple,
        call_kwargs: dict,
    ) -> "Context":
        """Return a new pending Context linked to this node and parents."""
        ctx = Context.__new__(Context)
        ctx._data = None  # Will be populated at execution
        ctx._parents = parents
        ctx._call_args = call_args
        ctx._call_kwargs = call_kwargs
        ctx._node = node
        ctx._executed = False
        ctx._executor = None
        ctx._context_factory = self._context_factory
        return ctx

    def _get_executor(self) -> DAGExecutor:
        if self._executor is None:
            self._executor = DAGExecutor(self, context_factory=self._context_factory)
        return self._executor

    async def _run_pending_async(self) -> None:
        if self._executed:
            return
        await self._get_executor().run_async()

    def _run_pending_sync(self) -> None:
        if self._executed:
            return
        self._get_executor().run_sync()

    @property
    def data(self) -> T:
        """Get the value, materializing pending nodes if needed."""
        self._run_pending_sync()
        return self._data

    async def _data_async(self) -> T:
        """Async get - materializes pending nodes first."""
        await self._run_pending_async()
        return self._data

    def __await__(self):
        """Awaiting a Context returns the materialized data."""
        return self._data_async().__await__()

    @property
    def is_pending(self) -> bool:
        """True if this context has unexecuted nodes."""
        return not self._executed

    @property
    def _pending_nodes(self) -> list["Context"]:
        """List of pending nodes in topological order."""
        from .graph import traverse_pending

        return traverse_pending(self)

    @property
    def pending_count(self) -> int:
        """Count of unexecuted nodes in the graph."""
        return len(self._pending_nodes)

    @property
    def pending_names(self) -> list[str]:
        """Names of unexecuted nodes in execution order."""
        return [ctx._node.name for ctx in self._pending_nodes]

    def __repr__(self) -> str:
        self._run_pending_sync()
        return f"Context({self._data!r})"
