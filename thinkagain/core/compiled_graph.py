"""
CompiledGraph - Immutable executable graph representation.

This is the result of calling Graph.compile(). It represents a graph
that has been validated, potentially optimized, and is ready for execution.

The compilation pattern separates graph construction from execution:
- Graph: Builder API for constructing graph structure
- CompiledGraph: Optimized, immutable executor

Benefits:
- Validate once at compile time, not on every execution
- Support multiple execution strategies (nested vs flat)
- Enable compile-time optimizations
- Clear separation of concerns
"""

from typing import Dict, Optional, Any
from .constants import END
from .context import Context
from .runtime import EdgeTarget, execute_graph

class CompiledGraph:
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
        is_flattened: bool = False,
    ):
        """
        Initialize compiled graph.

        Args:
            name: Graph name
            nodes: Mapping of node names to executables
            edges: Mapping of node names to edge targets
            entry_point: Starting node
            max_steps: Optional step limit
            is_flattened: Whether subgraphs have been inlined
        """
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.max_steps = max_steps
        self.is_flattened = is_flattened

        # Make immutable (shallow - prevents adding/removing nodes)
        self._sealed = True

    async def arun(self, ctx: Context) -> Context:
        """
        Execute the compiled graph.

        Args:
            ctx: Input context with initial state

        Returns:
            Context with execution results and path history

        Raises:
            ValueError: If execution fails
        """
        prefix = (
            f"[CompiledGraph:{self.name}]"
            if self.is_flattened
            else f"[Graph:{self.name}]"
        )
        return await execute_graph(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=prefix,
        )

    def visualize(self) -> str:
        """
        Generate Mermaid diagram syntax for the compiled graph.

        Returns:
            Mermaid diagram code
        """
        from .visualization import generate_mermaid_diagram

        return generate_mermaid_diagram(self.nodes, self.edges, self.entry_point)

    def __repr__(self) -> str:
        mode = "flat" if self.is_flattened else "nested"
        return (
            f"CompiledGraph(name='{self.name}', nodes={len(self.nodes)}, mode='{mode}')"
        )
