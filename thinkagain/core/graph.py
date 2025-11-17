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

import warnings
from collections import deque
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple

from .context import Context
from .executable import Executable
from .runtime import EdgeTarget, execute_graph


# Special constant for graph termination
END = "__end__"

RouteFn = Callable[[Context], str]
EdgePaths = Dict[str, str]


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
        self.edges: Dict[str, EdgeTarget] = {}  # node_name -> next | (route_fn, paths)
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
        self._ensure_node_exists(name)
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
        self._ensure_node_exists(from_node)
        self._assert_edge_available(from_node)
        self._assert_valid_target(to_node)
        self.edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self, from_node: str, route: RouteFn, paths: EdgePaths
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
        self._ensure_node_exists(from_node)
        self._assert_edge_available(from_node)
        normalized_paths = self._normalize_paths(paths)
        self.edges[from_node] = (route, normalized_paths)
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

        return await execute_graph(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=f"[Graph:{self.name}]",
        )

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

            edge = self.edges.get(current)
            if not edge:
                continue
            for target in self._edge_targets(edge):
                if target != END:
                    queue.append(target)

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
        1. Recursively expanding all Graph nodes into their constituent nodes
        2. Prefixing node names to avoid collisions (e.g., subgraph__worker)
        3. Adding virtual __END__ nodes for each subgraph
        4. Rewiring edges: subgraph's END references become subgraph__END__ nodes
        5. Parent edges connect to subgraph__END__ nodes, not internal nodes

        This approach treats END as a proper node, eliminating edge-merging complexity.

        Returns:
            CompiledGraph with flattened structure

        Raises:
            ValueError: If a cycle is detected in the graph hierarchy

        Example:
            # Before flattening:
            # outer: subgraph -> END
            # subgraph: worker -> (loop: worker, done: END)
            #
            # After flattening:
            # outer: subgraph__worker -> (loop: subgraph__worker, done: subgraph__END__)
            #        subgraph__END__ -> END
        """
        from .compiled_graph import CompiledGraph

        flattener = _GraphFlattener(self)
        flat_nodes, flat_edges, new_entry = flattener.flatten()

        if new_entry is None:
            raise ValueError("Entry point missing when flattening graph")

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

    def _ensure_node_exists(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

    def _assert_edge_available(self, name: str) -> None:
        if name in self.edges:
            raise ValueError(
                f"Node '{name}' already has an outgoing edge. "
                f"Use add_conditional_edge for multiple paths."
            )

    def _assert_valid_target(self, target: str, *, path_label: Optional[str] = None) -> None:
        if target == END:
            return
        if target not in self.nodes:
            if path_label is not None:
                raise ValueError(
                    f"Path '{path_label}' points to non-existent node '{target}'"
                )
            raise ValueError(f"Node '{target}' does not exist")

    def _normalize_paths(self, paths: EdgePaths) -> EdgePaths:
        normalized: EdgePaths = {}
        for label, target in paths.items():
            self._assert_valid_target(target, path_label=label)
            normalized[label] = target
        return normalized

    @staticmethod
    def _edge_targets(edge: EdgeTarget) -> Iterable[str]:
        if isinstance(edge, tuple):
            _, edge_map = edge
            return edge_map.values()
        return (edge,)


class _GraphFlattener:
    """
    Helper that rewrites a nested Graph into a flat node/edge map.

    Breaking the flattening logic into a dedicated helper keeps the main
    Graph class easier to scan while providing a single place to reason
    about recursion, prefix naming, and cycle detection.
    """

    def __init__(self, root: "Graph"):
        self.root = root
        self.flat_nodes: Dict[str, Any] = {}
        self.flat_edges: Dict[str, EdgeTarget] = {}
        self._visited: Set[int] = set()

    def flatten(self) -> Tuple[Dict[str, Any], Dict[str, EdgeTarget], Optional[str]]:
        node_mapping: Dict[str, Tuple[str, str]] = {}
        for node_name, node in self.root.nodes.items():
            node_mapping[node_name] = self._flatten_node(node_name, node)

        for from_node, edge in self.root.edges.items():
            _, from_exit = node_mapping[from_node]
            self.flat_edges[from_exit] = self._rewrite_edge(edge, node_mapping, END)

        entry = None
        if self.root.entry_point is not None:
            entry, _ = node_mapping[self.root.entry_point]

        return self.flat_nodes, self.flat_edges, entry

    def _flatten_node(
        self, node_name: str, node: Any, prefix: str = ""
    ) -> Tuple[str, str]:
        full_name = f"{prefix}{node_name}" if prefix else node_name
        if isinstance(node, Graph):
            return self._flatten_subgraph(full_name, node)

        self.flat_nodes[full_name] = node
        return full_name, full_name

    def _flatten_subgraph(self, full_name: str, graph: "Graph") -> Tuple[str, str]:
        graph_id = id(graph)
        if graph_id in self._visited:
            raise ValueError(
                f"Subgraph cycle detected: graph '{graph.name}' contains itself "
                f"directly or indirectly. Cannot flatten cyclic graph hierarchies."
            )

        self._visited.add(graph_id)
        try:
            if graph.entry_point is None:
                raise ValueError(
                    f"Subgraph '{graph.name}' is missing an entry point during flattening"
                )

            prefix = f"{full_name}__"
            virtual_end = f"{full_name}__END__"

            sub_mapping: Dict[str, Tuple[str, str]] = {}
            for sub_name, sub_node in graph.nodes.items():
                sub_mapping[sub_name] = self._flatten_node(sub_name, sub_node, prefix)

            for from_node, edge in graph.edges.items():
                _, from_exit = sub_mapping[from_node]
                self.flat_edges[from_exit] = self._rewrite_edge(
                    edge, sub_mapping, virtual_end
                )

            entry, _ = sub_mapping[graph.entry_point]
            return entry, virtual_end
        finally:
            self._visited.remove(graph_id)

    def _rewrite_edge(
        self,
        edge: EdgeTarget,
        mapping: Dict[str, Tuple[str, str]],
        default_target: str,
    ) -> EdgeTarget:
        if isinstance(edge, tuple):
            route_fn, paths = edge
            updated_paths: Dict[str, str] = {}
            for label, target in paths.items():
                if target == END:
                    updated_paths[label] = default_target
                else:
                    updated_paths[label] = mapping[target][0]
            return (route_fn, updated_paths)

        if edge == END:
            return default_target
        return mapping[edge][0]
