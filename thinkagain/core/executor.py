"""Execution logic for pending nodes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .errors import NodeExecutionError, NodeSignatureError

if TYPE_CHECKING:
    from .context import Context


class PendingExecutor:
    """Execute pending nodes for a Context."""

    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    async def run_async(self) -> None:
        ctx = self._ctx
        if not ctx._pending:
            return

        executed: list[str] = []
        # Copy metadata once at execution start (avoids copy overhead during chaining)
        metadata = ctx._metadata.copy()
        from .context import Context

        run_ctx = Context(ctx._data, metadata=metadata)

        for node in ctx._pending:
            try:
                start = time.perf_counter()
                run_ctx = await node.execute(run_ctx)
                duration = time.perf_counter() - start
                # Record on our tracked metadata, not run_ctx._metadata (node may have replaced it)
                metadata.record_node(node.name, duration)
                executed.append(node.name)
                # Check if node returned ctx with unawaited pending nodes
                if run_ctx._pending:
                    pending_names = [n.name for n in run_ctx._pending]
                    raise RuntimeError(
                        f"Node '{node.name}' returned context with unawaited pending nodes: {pending_names}. "
                        f"Use 'await ctx' before returning from a node that calls other nodes."
                    )
            except (NodeExecutionError, NodeSignatureError):
                raise
            except Exception as e:
                raise NodeExecutionError(node.name, executed, e) from e

        metadata.finished_at = time.time()
        object.__setattr__(ctx, "_data", run_ctx._data)
        object.__setattr__(ctx, "_pending", [])
        object.__setattr__(ctx, "_metadata", metadata)

    def run_sync(self) -> None:
        if not self._ctx._pending:
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
