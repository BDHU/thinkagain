"""Execution logic for pending nodes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .errors import NodeExecutionError
from .graph import traverse_pending
from .node import _ParentRef

if TYPE_CHECKING:
    from .context import Context


class DAGExecutor:
    """Execute pending nodes as a DAG with deduplication.

    Walks back-pointers from the target context to build the execution graph,
    then executes nodes in topological order. Shared ancestors are only
    executed once, enabling proper fanout handling.

    Note:
        Concurrent materialization of contexts that share ancestors is not
        supported. Do not use ``asyncio.gather()`` on related contexts.
        Instead, materialize them sequentially::

            # Wrong - undefined behavior:
            await asyncio.gather(ctx2.amaterialize(), ctx3.amaterialize())

            # Correct:
            await ctx2.amaterialize()
            await ctx3.amaterialize()
    """

    def __init__(self, ctx: "Context"):
        self._target = ctx
        self._results: dict[int, dict] = {}  # id(ctx) -> result data

    async def run_async(self) -> None:
        if self._target._executed:
            return

        # Collect all contexts by walking back-pointers (returns in topological order)
        execution_order = traverse_pending(self._target)
        if not execution_order:
            self._target._executed = True
            return

        executed_names: list[str] = []

        from .context import Context

        for ctx in execution_order:
            if ctx._executed:
                # Already executed (shared ancestor from another branch)
                self._results[id(ctx)] = ctx._data
                continue

            if ctx._node is None:
                # Root context (no node to execute)
                self._results[id(ctx)] = ctx._data
                object.__setattr__(ctx, "_executed", True)
                continue

            # Build execution args/kwargs by replacing ParentRef with materialized contexts
            def resolve_parent(value):
                if isinstance(value, _ParentRef):
                    parent = ctx._parents[value.index]
                    parent_data = self._results.get(id(parent), parent._data)
                    return Context(dict(parent_data))
                return value

            exec_args = [resolve_parent(arg) for arg in ctx._call_args]
            exec_kwargs = {
                key: resolve_parent(value) for key, value in ctx._call_kwargs.items()
            }

            try:
                result_ctx = await ctx._node.execute(*exec_args, **exec_kwargs)

                # Check if node returned context with unawaited pending nodes
                if result_ctx._node is not None and not result_ctx._executed:
                    pending = result_ctx.pending_names
                    raise RuntimeError(
                        f"Node '{ctx._node.name}' returned context with unawaited pending nodes: {pending}. "
                        f"Use 'await ctx' before returning from a node that calls other nodes."
                    )

                executed_names.append(ctx._node.name)

                # Cache result and update context
                self._results[id(ctx)] = result_ctx._data
                object.__setattr__(ctx, "_data", result_ctx._data)
                object.__setattr__(ctx, "_executed", True)

            except NodeExecutionError:
                raise
            except Exception as e:
                raise NodeExecutionError(ctx._node.name, executed_names, e) from e

    def run_sync(self) -> None:
        if self._target._executed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.run_async())
        else:
            raise RuntimeError(
                "Cannot materialize synchronously from async context. "
                "Use 'await ctx' or 'await arun(pipeline)' instead."
            )
