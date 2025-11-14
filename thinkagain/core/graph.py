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
from typing import Callable, Dict, Optional, Any, List
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

    def add_node(self, name: str, executable: Any) -> "Graph":
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

    def set_entry(self, name: str) -> "Graph":
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

    def add_edge(self, from_node: str, to_node: str) -> "Graph":
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
        self, from_node: str, route: Callable[[Context], str], paths: Dict[str, str]
    ) -> "Graph":
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
        descriptor = (
            f"Entering subgraph: {node_name} ({node.name})"
            if isinstance(node, Graph)
            else f"Executing: {node_name}"
        )
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
            raise ValueError(
                "Entry point not set. Use set_entry() or add the first node."
            )

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
                lines.append(f'    {node_name}[["{label}"]]')
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
        if hasattr(node, "arun"):
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
                fn_name = (
                    route_fn.__name__ if hasattr(route_fn, "__name__") else "lambda"
                )
                edges_dict[k] = {
                    "type": "conditional",
                    "function": fn_name,
                    "paths": edge_map,
                }
            else:
                edges_dict[k] = {"type": "direct", "to": v}

        return {
            "type": "Graph",
            "name": self.name,
            "entry_point": self.entry_point,
            "max_steps": self.max_steps,
            "nodes": {
                name: node.to_dict()
                if hasattr(node, "to_dict")
                else {"type": "callable", "name": str(node)}
                for name, node in self.nodes.items()
            },
            "edges": edges_dict,
        }

    def compile(self, flatten: bool = False) -> "CompiledGraph":
        """
        Compile graph into executable representation.

        This validates the graph structure and returns an optimized,
        immutable executor. Compilation happens once; execution can
        happen many times.

        Args:
            flatten: If True, recursively inline all subgraphs into a
                    flat structure. If False, keep subgraphs as black boxes.

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValueError: If graph structure is invalid

        Example:
            # Build graph
            graph = Graph()
            graph.add_node("a", worker_a)
            graph.add_node("b", subgraph_b)
            graph.add_edge("a", "b")

            # Compile with nested subgraphs (black box)
            nested = graph.compile(flatten=False)
            result = await nested.arun(ctx)

            # Compile with flattened subgraphs (all nodes visible)
            flat = graph.compile(flatten=True)
            result = await flat.arun(ctx)

            # Both execute correctly, just different structure
        """
        # Validate first
        self._validate()

        if flatten:
            return self._compile_flat()
        else:
            return self._compile_nested()

    def _compile_nested(self) -> "CompiledGraph":
        """
        Compile graph keeping subgraphs as black boxes.

        This is the simpler compilation path - just wrap the graph
        in a CompiledGraph executor without structural changes.

        Returns:
            CompiledGraph with nested subgraph structure
        """
        from .compiled_graph import CompiledGraph

        # Simple wrapper - no transformation needed
        return CompiledGraph(
            name=self.name,
            nodes=self.nodes.copy(),  # Shallow copy
            edges=self.edges.copy(),  # Shallow copy
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            is_flattened=False,
        )

    def _compile_flat(self) -> "CompiledGraph":
        """
        Compile graph with all subgraphs recursively inlined.

        This flattens the graph structure by:
        1. Recursively expanding all Graph nodes
        2. Prefixing node names to avoid collisions
        3. Rewriting conditional edges with new names
        4. Creating a single flat execution graph

        Returns:
            CompiledGraph with flattened structure

        Raises:
            ValueError: If a cycle is detected in the graph hierarchy

        Example:
            # Before flattening:
            # graph: a -> subgraph(x -> y) -> b
            #
            # After flattening:
            # graph: a -> subgraph__x -> subgraph__y -> b
        """
        from .compiled_graph import CompiledGraph

        flat_nodes = {}
        flat_edges = {}
        new_entry = None

        # Process nodes in topological order (roughly)
        def flatten_node(
            node_name: str, node: Any, prefix: str = "", visited_graphs: set = None
        ) -> List[str]:
            """
            Flatten a single node, returning list of actual node names added.

            Args:
                node_name: Original node name
                node: The executable
                prefix: Prefix for naming
                visited_graphs: Set of graph IDs currently being processed (for cycle detection)

            Returns:
                List of node names added to flat_nodes

            Raises:
                ValueError: If a subgraph cycle is detected
            """
            if visited_graphs is None:
                visited_graphs = set()

            full_name = f"{prefix}{node_name}" if prefix else node_name

            # If it's a Graph, recursively flatten it
            if isinstance(node, Graph):
                # Check for cycle: is this graph already being processed?
                graph_id = id(node)
                if graph_id in visited_graphs:
                    raise ValueError(
                        f"Subgraph cycle detected: graph '{node.name}' contains itself "
                        f"directly or indirectly. Cannot flatten cyclic graph hierarchies."
                    )

                # Add this graph to visited set before recursing
                visited_graphs.add(graph_id)

                try:
                    added_nodes = []
                    subgraph_prefix = f"{full_name}__"

                    # Flatten all subgraph nodes
                    for sub_node_name, sub_node in node.nodes.items():
                        sub_added = flatten_node(
                            sub_node_name, sub_node, subgraph_prefix, visited_graphs
                        )
                        added_nodes.extend(sub_added)

                    # Rewrite edges from subgraph
                    for from_node, edge in node.edges.items():
                        prefixed_from = f"{subgraph_prefix}{from_node}"

                        if isinstance(edge, tuple):
                            # Conditional edge - rewrite paths
                            route_fn, paths = edge
                            new_paths = {}
                            for key, to_node in paths.items():
                                if to_node == END:
                                    new_paths[key] = END
                                else:
                                    new_paths[key] = f"{subgraph_prefix}{to_node}"
                            flat_edges[prefixed_from] = (route_fn, new_paths)
                        else:
                            # Direct edge
                            if edge == END:
                                flat_edges[prefixed_from] = END
                            else:
                                flat_edges[prefixed_from] = f"{subgraph_prefix}{edge}"

                    return added_nodes
                finally:
                    # Remove this graph from visited set after processing
                    visited_graphs.discard(graph_id)

            else:
                # Regular node - just add it
                flat_nodes[full_name] = node
                return [full_name]

        # Flatten all top-level nodes
        node_mapping = {}  # original name -> list of flattened names
        for node_name, node in self.nodes.items():
            flattened = flatten_node(node_name, node)
            node_mapping[node_name] = flattened

        # Rewrite top-level edges
        for from_node, edge in self.edges.items():
            # Get the LAST node from the flattened "from" node
            # (the exit point of that subgraph, or the node itself)
            from_names = node_mapping[from_node]
            actual_from = from_names[-1] if from_names else from_node

            if isinstance(edge, tuple):
                # Conditional edge
                route_fn, paths = edge
                new_paths = {}
                for key, to_node in paths.items():
                    if to_node == END:
                        new_paths[key] = END
                    else:
                        # Get FIRST node from flattened "to" node (entry point)
                        to_names = node_mapping.get(to_node, [to_node])
                        new_paths[key] = to_names[0] if to_names else to_node
                flat_edges[actual_from] = (route_fn, new_paths)
            else:
                # Direct edge
                if edge == END:
                    flat_edges[actual_from] = END
                else:
                    to_names = node_mapping.get(edge, [edge])
                    flat_edges[actual_from] = to_names[0] if to_names else edge

        # Determine new entry point (first node of original entry)
        if self.entry_point:
            entry_names = node_mapping.get(self.entry_point, [self.entry_point])
            new_entry = entry_names[0] if entry_names else self.entry_point

        return CompiledGraph(
            name=f"{self.name}_flat",
            nodes=flat_nodes,
            edges=flat_edges,
            entry_point=new_entry,
            max_steps=self.max_steps,
            is_flattened=True,
        )

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, entry='{self.entry_point}')"
