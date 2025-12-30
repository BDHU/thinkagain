"""Execution logic for pending nodes."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Callable, ContextManager

from .errors import NodeExecutionError
from .graph import traverse_pending
from .node import _ParentRef

if TYPE_CHECKING:
    from .context import Context

# Thread-local event loop cache for run_sync()
_thread_local = threading.local()


class DAGExecutor:
    """Execute pending nodes as a DAG with deduplication.

    Walks back-pointers from the target context to build the execution graph,
    then executes nodes in topological order. Shared ancestors are only
    executed once, enabling proper fanout handling.

    Note:
        Concurrent materialization of contexts that share ancestors is not
        supported. Do not use ``asyncio.gather()`` on related contexts.
        Instead, materialize them sequentially.
    """

    def __init__(
        self,
        ctx: "Context",
        context_factory: Callable[[str], ContextManager[None]] | None = None,
    ):
        self._target = ctx
        self._results: dict[int, "Context"] = {}  # id(ctx) -> executed context
        self._context_factory = context_factory

    def _resolve_parent(self, ctx: "Context", value):
        if isinstance(value, _ParentRef):
            parent = ctx._parents[value.index]
            return self._results.get(id(parent), parent)
        return value

    async def run_async(self) -> None:
        if self._target._executed:
            return

        # Collect all contexts by walking back-pointers (returns in topological order)
        execution_order = traverse_pending(self._target)
        if not execution_order:
            object.__setattr__(self._target, "_executed", True)
            return

        executed_names: list[str] = []

        for ctx in execution_order:
            if ctx._executed:
                # Already executed (shared ancestor from another branch)
                self._results[id(ctx)] = ctx
                continue

            if ctx._node is None:
                # Root context (no node to execute)
                self._results[id(ctx)] = ctx
                object.__setattr__(ctx, "_executed", True)
                continue

            # Build execution args/kwargs by replacing ParentRef with materialized contexts
            exec_args = [self._resolve_parent(ctx, arg) for arg in ctx._call_args]
            exec_kwargs = {
                key: self._resolve_parent(ctx, value)
                for key, value in ctx._call_kwargs.items()
            }

            try:
                if self._context_factory is not None:
                    with self._context_factory(ctx._node.name):
                        result_ctx = await ctx._node.execute(*exec_args, **exec_kwargs)
                else:
                    result_ctx = await ctx._node.execute(*exec_args, **exec_kwargs)

                executed_names.append(ctx._node.name)

                # Store result data and mark as executed
                object.__setattr__(ctx, "_data", result_ctx._data)
                object.__setattr__(ctx, "_executed", True)
                self._results[id(ctx)] = ctx

            except Exception as e:
                if isinstance(e, NodeExecutionError):
                    raise
                raise NodeExecutionError(ctx._node.name, executed_names, e) from e

    def run_sync(self) -> None:
        if self._target._executed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Reuse cached event loop to avoid asyncio.run() overhead
            loop = getattr(_thread_local, "loop", None)
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                _thread_local.loop = loop
            loop.run_until_complete(self.run_async())
        else:
            raise RuntimeError(
                "Cannot materialize synchronously from async context. "
                "Use 'await ctx' instead."
            )
