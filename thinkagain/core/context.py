"""Context with lazy execution support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node import Node


class NodeSignatureError(TypeError):
    """Raised when a node function has an invalid signature."""

    def __init__(self, node_name: str, cause: TypeError):
        super().__init__(
            f"Node '{node_name}' has invalid signature. "
            f"Nodes must take exactly one parameter (ctx). Cause: {cause}"
        )


class NodeExecutionError(Exception):
    """Raised when a node in a pipeline fails during execution.

    Attributes:
        node_name: Name of the node where the error occurred.
        executed: Names of nodes that completed successfully before the failure.
        cause: The original exception raised by the failing node.
    """

    def __init__(self, node_name: str, executed: list[str], cause: Exception):
        self.node_name = node_name
        self.executed: tuple[str, ...] = tuple(executed)
        self.cause = cause
        executed_display = ", ".join(self.executed) if self.executed else "none"
        super().__init__(
            f"Node '{node_name}' failed after executing: {executed_display}. Cause: {cause!r}"
        )


class Context:
    """Dict-like context with lazy execution. Data is shared by default."""

    __slots__ = ("_data", "_pending")

    def __init__(self, data: Context | dict | None = None):
        if isinstance(data, Context):
            # Copy from another Context (always copies data for isolation)
            object.__setattr__(self, "_data", dict(data._data))
            object.__setattr__(self, "_pending", list(data._pending))
        else:
            object.__setattr__(self, "_data", data if data is not None else {})
            object.__setattr__(self, "_pending", [])

    def _chain(self, node: Node) -> Context:
        ctx = Context(self._data)  # shares data
        object.__setattr__(ctx, "_pending", self._pending + [node])
        return ctx

    def copy(self) -> Context:
        """Create a copy with isolated data."""
        return Context(self)

    async def _amaterialize(self) -> None:
        if not self._pending:
            return

        executed: list[str] = []
        ctx = Context(self._data)

        for node in self._pending:
            try:
                ctx = await node.execute(ctx)
                executed.append(node.name)
            except (NodeExecutionError, NodeSignatureError):
                raise
            except Exception as e:
                raise NodeExecutionError(node.name, executed, e) from e

        object.__setattr__(self, "_data", ctx._data)
        object.__setattr__(self, "_pending", [])

    def _materialize(self) -> None:
        if not self._pending:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._amaterialize())
        else:
            raise RuntimeError(
                "Cannot materialize synchronously from async context. "
                "Use 'await ctx' or 'await arun(pipeline)' instead."
            )

    def materialize(self) -> Context:
        """Synchronously execute all pending nodes and return this context."""
        self._materialize()
        return self

    async def amaterialize(self) -> Context:
        """Asynchronously execute all pending nodes and return this context."""
        await self._amaterialize()
        return self

    def __await__(self):
        """Awaiting a Context is equivalent to calling `amaterialize`."""
        return self.amaterialize().__await__()

    def __getattr__(self, name: str) -> Any:
        self._materialize()
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._materialize()
        self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        self._materialize()
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._materialize()
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        self._materialize()
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        self._materialize()
        return self._data.get(key, default)

    def peek(self, key: str, default: Any = None) -> Any:
        """Get value without materializing."""
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
        self._materialize()
        return list(self._data.keys())

    def values(self) -> list[Any]:
        self._materialize()
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        self._materialize()
        return list(self._data.items())

    def to_dict(self) -> dict:
        """Return a shallow copy of the underlying data."""
        self._materialize()
        return dict(self._data)

    def __iter__(self):
        self._materialize()
        return iter(self._data)

    def __repr__(self) -> str:
        self._materialize()
        return f"Context({self._data})"

    def __len__(self) -> int:
        self._materialize()
        return len(self._data)
