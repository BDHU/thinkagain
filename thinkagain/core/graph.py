"""
Graph-based execution supporting arbitrary cycles and conditional routing.

Provides Graph class for complex workflows with dynamic routing.
Everything is a Graph - Pipelines are just syntactic sugar.

Use Graph when you need:
- Cycles (loops back to previous nodes)
- Dynamic routing based on runtime state
- Multi-agent interactions with complex flow
- Subgraph composition (graphs within graphs)

For simple sequential flows, use the >> operator:
    pipeline = worker1 >> worker2 >> worker3

Example - Self-correcting RAG with cycle:
    graph = Graph(name="self_correcting_rag")
    graph.add_node("retrieve", retrieve_worker)
    graph.add_node("critique", critique_worker)
    graph.add_node("refine", refine_worker)
    graph.add_node("generate", generate_worker)

    graph.set_entry("retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edge(
        "critique",
        route=lambda ctx: "refine" if ctx.quality < 0.8 else "generate",
        paths={"refine": "refine", "generate": "generate"}
    )
    graph.add_edge("refine", "retrieve")  # Cycle!
    graph.add_edge("generate", END)

    result = await graph.arun(ctx)
"""

import asyncio
from collections import deque
from typing import Callable, Dict, Optional, Any
from .context import Context
from .executable import Executable, run_sync
import warnings


# Special constant for graph termination
END = "__end__"


class Graph(Executable):
    """
    Async-first graph supporting arbitrary cycles and conditional routing.

    Graphs can contain:
    - Workers (leaf computations)
    - Other Graphs (subgraphs)
    - Pipelines (sequential graphs)
    - Any Executable

    All composition is natural - just add executables as nodes.
    """

    def __init__(self, name: str = "graph", max_steps: Optional[int] = None):
        """
        Initialize a new graph.

        Args:
            name: Name for this graph (used in logging)
            max_steps: Optional maximum execution steps to prevent infinite loops.
                      If None (default), no limit is enforced.
        """
        super().__init__(name)
        self.nodes: Dict[str, Any] = {}  # node_name -> executable
        self.edges: Dict[str, Any] = {}  # node_name -> next | (route_fn, paths)
        self.entry_point: Optional[str] = None
        self.max_steps = max_steps

    def add_node(self, name: str, executable: Any) -> 'Graph':
        """
        Add a node to the graph.

        The node can be:
        - A Worker instance
        - Another Graph (subgraph)
        - A Pipeline (sequential subgraph)
        - Any callable that transforms Context

        Args:
            name: Unique identifier for this node
            executable: Executable to run at this node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node name already exists

        Example:
            # Add various types of nodes
            graph.add_node("worker", MyWorker())
            graph.add_node("subgraph", another_graph)
            graph.add_node("pipeline", worker1 >> worker2)
            graph.add_node("custom", lambda ctx: ctx)
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")

        self.nodes[name] = executable

        # Auto-set entry point if this is the first node
        if self.entry_point is None:
            self.entry_point = name

        return self

    def set_entry(self, name: str) -> 'Graph':
        """
        Set the starting node for graph execution.

        Args:
            name: Name of the entry node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node does not exist
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")
        self.entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> 'Graph':
        """
        Add a direct edge between two nodes.

        Args:
            from_node: Source node name
            to_node: Destination node name (or END to terminate)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If nodes don't exist or from_node already has an edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' does not exist")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' does not exist")

        if from_node in self.edges:
            raise ValueError(
                f"Node '{from_node}' already has an outgoing edge. "
                f"Use add_conditional_edge for multiple paths."
            )

        self.edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        route: Callable[[Context], str],
        paths: Dict[str, str]
    ) -> 'Graph':
        """
        Add a conditional edge that routes based on context state.

        The route function examines the context and returns a key from
        the paths dict, which maps to the next node.

        Args:
            from_node: Source node name
            route: Function that takes Context and returns a path key
            paths: Mapping of route keys to node names

        Returns:
            Self for method chaining

        Example:
            graph.add_conditional_edge(
                "critique",
                route=lambda ctx: "high" if ctx.score > 0.8 else "low",
                paths={"high": "generate", "low": "refine"}
            )

        Raises:
            ValueError: If from_node doesn't exist or already has an edge
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' does not exist")

        if from_node in self.edges:
            raise ValueError(f"Node '{from_node}' already has an outgoing edge")

        # Validate all path destinations exist
        for path_key, to_node in paths.items():
            if to_node != END and to_node not in self.nodes:
                raise ValueError(
                    f"Path '{path_key}' points to non-existent node '{to_node}'"
                )

        self.edges[from_node] = (route, paths)
        return self

    async def arun(self, ctx: Context) -> Context:
        """
        Execute the graph starting from entry point (async).

        This is the primary execution method. For sync execution,
        use __call__() which wraps this in asyncio.run().

        Args:
            ctx: Input context with initial state

        Returns:
            Context with execution results and path history

        Raises:
            ValueError: If graph is invalid or execution fails
        """
        # Lazy validation on first execution
        self._validate()

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

        Automatically handles:
        - Workers
        - Subgraphs (nested graphs)
        - Pipelines (sequential graphs)
        - Plain callables

        Args:
            node_name: Name of node to execute
            ctx: Current context

        Returns:
            Updated context after node execution
        """
        node = self.nodes[node_name]
        descriptor = f"Entering subgraph: {node_name} ({node.name})" if isinstance(node, Graph) else f"Executing: {node_name}"
        self._log(ctx, descriptor)

        try:
            return await self._invoke(node, ctx)
        except Exception as e:
            self._log(ctx, f"Error in node '{node_name}': {e}")
            raise

    async def _resolve_next_node(self, current: str, ctx: Context) -> Optional[str]:
        """
        Resolve the next node based on edges.

        Handles both direct edges and conditional routing.

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

    def _validate(self):
        """Validate graph structure (called lazily on first execution)."""
        if self.entry_point is None:
            raise ValueError("Entry point not set. Use set_entry() or add the first node.")

        # Detect unreachable nodes (warning only)
        reachable = self._find_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            warnings.warn(f"Unreachable nodes detected: {unreachable}")

        # Detect nodes without outgoing edges (warning only)
        dead_ends = [node for node in self.nodes if node not in self.edges]
        if dead_ends:
            warnings.warn(
                f"Nodes without outgoing edges: {dead_ends}. "
                f"Consider adding edges to END."
            )

    def _find_reachable_nodes(self) -> set:
        """BFS to find all reachable nodes from entry point."""
        reachable = set()
        queue = deque([self.entry_point])

        while queue:
            current = queue.popleft()
            if current in reachable or current == END:
                continue

            reachable.add(current)

            # Find outgoing edges
            if current in self.edges:
                edge = self.edges[current]
                if isinstance(edge, tuple):  # Conditional edge
                    _, edge_map = edge
                    queue.extend(v for v in edge_map.values() if v != END)
                else:  # Direct edge
                    if edge != END:
                        queue.append(edge)

        return reachable

    def visualize(self) -> str:
        """
        Generate Mermaid diagram syntax for the graph.

        Returns:
            Mermaid diagram code that can be rendered or saved

        Example:
            print(graph.visualize())
            # Copy output to mermaid.live or GitHub markdown
        """
        lines = ["graph TD", f"    START([START]) --> {self.entry_point}"]

        for node_name, node in self.nodes.items():
            if isinstance(node, Graph):
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
        ctx.log(f"[Graph:{self.name}] {message}")

    def to_dict(self) -> dict:
        """Export graph structure as dictionary."""
        edges_dict = {}
        for k, v in self.edges.items():
            if isinstance(v, tuple):
                route_fn, edge_map = v
                fn_name = route_fn.__name__ if hasattr(route_fn, '__name__') else "lambda"
                edges_dict[k] = {
                    "type": "conditional",
                    "function": fn_name,
                    "paths": edge_map
                }
            else:
                edges_dict[k] = {"type": "direct", "to": v}

        return {
            "type": "Graph",
            "name": self.name,
            "entry_point": self.entry_point,
            "max_steps": self.max_steps,
            "nodes": {
                name: node.to_dict() if hasattr(node, 'to_dict')
                else {"type": "callable", "name": str(node)}
                for name, node in self.nodes.items()
            },
            "edges": edges_dict
        }

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, entry='{self.entry_point}')"
