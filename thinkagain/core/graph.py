"""
Graph-based execution supporting arbitrary cycles and conditional routing.

Provides Graph class for complex workflows with dynamic routing.
Everything is a Graph - the >> operator creates sequential graphs automatically.

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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

from .constants import END
from .context import Context
from .graph_executor import GraphExecutor
from .runtime import EdgeTarget, DirectEdge, ConditionalEdge
from .graph_flattener import GraphFlattener

if TYPE_CHECKING:
    from .compiled_graph import CompiledGraph

RouteFn = Callable[[Context], str]
EdgePaths = Dict[str, str]


class Graph(GraphExecutor):
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
        super().__init__(
            name=name,
            nodes={},
            edges={},
            entry_point=None,
            max_steps=max_steps,
        )

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
        self.edges[from_node] = DirectEdge(target=to_node)
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
        self.edges[from_node] = ConditionalEdge(route_fn=route, paths=normalized_paths)
        return self

    def _validate_before_run(self) -> None:
        super()._validate_before_run()
        self._validate()

    def _log_prefix(self) -> str:
        return f"[Graph:{self.name}]"

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
        from .visualization import generate_mermaid_diagram

        return generate_mermaid_diagram(self.nodes, self.edges, self.entry_point)

    def compile(self) -> "CompiledGraph":
        """
        Compile graph into an optimized, flattened representation.

        This validates the graph structure and returns an immutable
        executor with all subgraphs inlined. Build graphs once, compile,
        and then execute the compiled artifact many times.

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

            # Compile for execution (flattened view)
            flat = graph.compile()
            result = await flat.arun(ctx)
        """
        # Validate first
        self._validate()

        return self._compile_flat()

    def _compile_flat(self) -> "CompiledGraph":
        """
        Helper that compiles the graph with all subgraphs recursively inlined.

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

        flattener = GraphFlattener(self)
        flat_nodes, flat_edges, new_entry = flattener.flatten()

        if new_entry is None:
            raise ValueError("Entry point missing when flattening graph")

        return CompiledGraph(
            name=f"{self.name}_flat",
            nodes=flat_nodes,
            edges=flat_edges,
            entry_point=new_entry,
            max_steps=self.max_steps,
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

    def _assert_valid_target(
        self, target: str, *, path_label: Optional[str] = None
    ) -> None:
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
        if isinstance(edge, ConditionalEdge):
            return edge.paths.values()
        if isinstance(edge, DirectEdge):
            return (edge.target,)
        # Backward compat: plain string
        return (edge,)
