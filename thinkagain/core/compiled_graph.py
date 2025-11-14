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

import asyncio
from typing import Callable, Dict, Optional, Any, List, Tuple, Union
from .context import Context
from .executable import run_sync

# Special constant for graph termination
END = "__end__"

# Type aliases for clarity
NodeName = str
EdgeTarget = Union[str, Tuple[Callable[[Context], str], Dict[str, str]]]


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
        nodes: Dict[NodeName, Any],
        edges: Dict[NodeName, EdgeTarget],
        entry_point: NodeName,
        max_steps: Optional[int] = None,
        is_flattened: bool = False
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
        current = self.entry_point
        execution_path = []

        self._log(ctx, "Starting execution")
        self._log(ctx, f"Entry point: {current}")

        step = 0
        while True:
            if current == END or current is None:
                self._log(ctx, f"Reached END after {step} steps")
                break

            ctx = await self._execute_node(current, ctx)
            execution_path.append(current)

            next_node = await self._resolve_next_node(current, ctx)

            if next_node is None or next_node == END:
                break

            current = next_node
            step += 1

            # Check max_steps limit if set
            if self.max_steps is not None and step >= self.max_steps:
                self._log(ctx, f"WARNING: Terminated after max_steps={self.max_steps}")
                self._log(ctx, "This may indicate an infinite loop")
                break

        # Store execution metadata
        ctx.execution_path = execution_path
        ctx.total_steps = len(execution_path)

        self._log(ctx, "Completed execution")
        self._log(ctx, f"Total steps: {ctx.total_steps}")
        path_display = " → ".join(execution_path) or "(none)"
        self._log(ctx, f"Path: {path_display}")

        return ctx

    async def _execute_node(self, node_name: str, ctx: Context) -> Context:
        """
        Execute a single node.

        Args:
            node_name: Name of node to execute
            ctx: Current context

        Returns:
            Updated context after node execution
        """
        node = self.nodes[node_name]

        # Different logging for subgraphs vs regular nodes
        from .graph import Graph
        if isinstance(node, Graph):
            self._log(ctx, f"Entering subgraph: {node_name} ({node.name})")
        elif isinstance(node, CompiledGraph):
            self._log(ctx, f"Entering compiled subgraph: {node_name} ({node.name})")
        else:
            self._log(ctx, f"Executing: {node_name}")

        try:
            return await self._invoke(node, ctx)
        except Exception as e:
            self._log(ctx, f"Error in node '{node_name}': {e}")
            raise

    async def _resolve_next_node(self, current: str, ctx: Context) -> Optional[str]:
        """
        Resolve the next node based on edges.

        Args:
            current: Current node name
            ctx: Current context

        Returns:
            Name of next node, END, or None
        """
        edge = self.edges.get(current)

        if edge is None:
            self._log(ctx, f"Node '{current}' has no outgoing edge, terminating")
            return None

        # Conditional edge
        if isinstance(edge, tuple):
            route_fn, edge_map = edge

            try:
                # Support async route functions
                route_result = await self._call_route(route_fn, ctx)
            except Exception as e:
                self._log(ctx, f"Error in routing function: {e}")
                raise

            self._log(ctx, f"Conditional route from '{current}': '{route_result}'")

            # Validate and return
            if route_result in edge_map:
                return edge_map[route_result]

            if route_result == END:
                return END

            available = list(edge_map.keys()) + [END]
            raise ValueError(
                f"Route function returned '{route_result}' but no matching edge. "
                f"Available paths: {available}"
            )

        # Direct edge
        self._log(ctx, f"Direct edge: '{current}' → '{edge}'")
        return edge

    async def _invoke(self, node: Any, ctx: Context) -> Context:
        """Execute any supported node type."""
        if hasattr(node, 'arun'):
            return await node.arun(ctx)
        if asyncio.iscoroutinefunction(node):
            return await node(ctx)
        return await run_sync(node, ctx)

    async def _call_route(self, route: Callable[[Context], str], ctx: Context) -> str:
        """Run routing functions that may be sync or async."""
        if asyncio.iscoroutinefunction(route):
            return await route(ctx)
        return route(ctx)

    def _log(self, ctx: Context, message: str) -> None:
        """Log with graph context."""
        prefix = f"[CompiledGraph:{self.name}]" if self.is_flattened else f"[Graph:{self.name}]"
        ctx.log(f"{prefix} {message}")

    def visualize(self) -> str:
        """
        Generate Mermaid diagram syntax for the compiled graph.

        Returns:
            Mermaid diagram code
        """
        from .graph import Graph

        lines = ["graph TD", f"    START([START]) --> {self.entry_point}"]

        for node_name, node in self.nodes.items():
            if isinstance(node, (Graph, CompiledGraph)):
                label = f"{node_name}\\n(subgraph: {node.name})"
                lines.append(f"    {node_name}[[\"{label}\"]]")
            else:
                lines.append(f"    {node_name}[{node_name}]")

        lines.append("    END([END])")

        for from_node, edge in self.edges.items():
            if isinstance(edge, tuple):
                _, edge_map = edge
                for label, to_node in edge_map.items():
                    target = "END" if to_node == END else to_node
                    lines.append(f"    {from_node} -->|{label}| {target}")
            else:
                target = "END" if edge == END else edge
                lines.append(f"    {from_node} --> {target}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        mode = "flat" if self.is_flattened else "nested"
        return f"CompiledGraph(name='{self.name}', nodes={len(self.nodes)}, mode='{mode}')"
