"""
Shared execution utilities for graph-like structures.

Graph and CompiledGraph both end up delegating to the same runtime module.
Historically each class reimplemented the same ``arun``, ``stream`` and
``visualize`` helpers which made the code harder to follow.  This module
centralizes the shared mechanics in ``GraphExecutor`` so individual graph
implementations can focus exclusively on how they are built (mutable Graph)
or stored (immutable CompiledGraph).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from .constants import END
from .context import Context
from .executable import Executable
from .runtime import EdgeTarget, StreamEvent, execute_graph, stream_graph_events
from .visualization import generate_mermaid_diagram


class GraphExecutor(Executable):
    """
    Base class that wires graph structures into the runtime helpers.

    Subclasses provide the concrete node/edge mappings and may override
    ``_validate_before_run`` or ``_log_prefix`` but everything else is handled
    centrally here.
    """

    def __init__(
        self,
        *,
        name: str,
        nodes: Dict[str, Any],
        edges: Dict[str, EdgeTarget],
        entry_point: Optional[str],
        max_steps: Optional[int] = None,
    ):
        super().__init__(name)
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.max_steps = max_steps

    async def arun(self, ctx: Context) -> Context:
        """Execute the graph using the shared runtime helpers."""
        self._validate_before_run()
        entry = self._ensure_entry_point()
        return await execute_graph(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=entry,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=self._log_prefix(),
        )

    async def stream(self, ctx: Context) -> AsyncIterator[StreamEvent]:
        """Yield ``StreamEvent`` objects as each node executes."""
        self._validate_before_run()
        entry = self._ensure_entry_point()
        async for event in stream_graph_events(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=entry,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=self._log_prefix(),
        ):
            yield event

    def visualize(self) -> str:
        """Generate Mermaid diagram syntax for the graph."""
        entry = self._ensure_entry_point()
        return generate_mermaid_diagram(self.nodes, self.edges, entry)

    def _ensure_entry_point(self) -> str:
        if self.entry_point is None:
            raise ValueError("Entry point not set. Use set_entry() before execution.")
        return self.entry_point

    def _validate_before_run(self) -> None:
        """Hook for subclasses to run validations prior to execution."""
        self._ensure_entry_point()

    def _log_prefix(self) -> str:
        """Text inserted before runtime log messages."""
        return f"[Graph:{self.name}]"


__all__ = ["GraphExecutor"]
