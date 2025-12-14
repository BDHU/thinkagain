"""
CompiledGraph - Immutable executable graph representation.

This is the result of calling Graph.compile(). It represents a graph
that has been validated and is ready for execution.

The compilation pattern separates graph construction from execution:
- Graph: Builder API for constructing graph structure
- CompiledGraph: Immutable executor

Benefits:
- Validate once at compile time, not on every execution
- Clear separation of concerns
- Can cache predecessor maps and other derived data
"""

from typing import Any, AsyncIterator, Dict, Optional

from .constants import END
from .context import Context
from .executable import Executable
from .runtime import EdgeMap, StreamEvent, execute_graph, stream_graph_events


class CompiledGraph(Executable):
    """
    Immutable, executable graph representation.

    This is created by Graph.compile() and should not be instantiated directly.
    All validation happens at compile time.

    Example:
        graph = Graph()
        graph.add("a", worker_a)
        graph.add("b", worker_b)
        graph.edge("a", "b")

        # Compile for execution
        compiled = graph.compile()

        # Execute multiple times
        result1 = await compiled.arun(ctx1)
        result2 = await compiled.arun(ctx2)
    """

    def __init__(
        self,
        name: str,
        nodes: Dict[str, Any],
        edges: EdgeMap,
        entry_point: str,
        max_steps: Optional[int] = None,
    ):
        """
        Initialize compiled graph.

        Args:
            name: Graph name
            nodes: Mapping of node names to executables
            edges: Mapping of node names to edge targets
            entry_point: Starting node
            max_steps: Optional step limit
        """
        super().__init__(name)
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.max_steps = max_steps

    async def arun(self, ctx: Context) -> Context:
        """Execute the graph and return the final context."""
        return await execute_graph(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=f"[{self.name}]",
        )

    async def stream(self, ctx: Context) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvent objects as each node executes."""
        async for event in stream_graph_events(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=f"[{self.name}]",
        ):
            yield event

    def visualize(self) -> str:
        """Generate Mermaid diagram syntax for the graph."""
        from .visualization import generate_mermaid_diagram

        return generate_mermaid_diagram(self.nodes, self.edges, self.entry_point)

    def __repr__(self) -> str:
        return f"CompiledGraph(name='{self.name}', nodes={len(self.nodes)})"
