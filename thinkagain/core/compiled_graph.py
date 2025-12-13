"""
CompiledGraph - Immutable executable graph representation.

This is the result of calling Graph.compile(). It represents a graph
that has been validated, potentially optimized, and is ready for execution.

The compilation pattern separates graph construction from execution:
- Graph: Builder API for constructing graph structure
- CompiledGraph: Optimized, immutable executor

Benefits:
- Validate once at compile time, not on every execution
- Enable compile-time optimizations
- Clear separation of concerns
"""

from typing import Any, Dict, Optional

from .graph_executor import GraphExecutor
from .runtime import EdgeTarget


class CompiledGraph(GraphExecutor):
    """
    Immutable, executable graph representation.

    This is created by Graph.compile() and should not be instantiated directly.
    All validation and optimization happens at compile time.

    Example:
        graph = Graph()
        graph.add_node("a", worker_a)
        graph.add_node("b", worker_b)
        graph.add_edge("a", "b")

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
        edges: Dict[str, EdgeTarget],
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
        super().__init__(
            name=name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            max_steps=max_steps,
        )
        # Make immutable (shallow - prevents adding/removing nodes)
        self._sealed = True

    def _log_prefix(self) -> str:
        return f"[CompiledGraph:{self.name}]"

    def __repr__(self) -> str:
        return f"CompiledGraph(name='{self.name}', nodes={len(self.nodes)})"
