"""
Graph-based execution supporting arbitrary cycles, conditional routing, and parallel execution.

Provides Graph class for complex workflows with dynamic routing.
Everything is a Graph - the >> operator creates sequential graphs automatically.

Use Graph when you need:
- Cycles (loops back to previous nodes)
- Dynamic routing based on runtime state
- Multi-agent interactions with complex flow
- Subgraph composition (graphs within graphs)
- Fan-out/fan-in parallel execution

For simple sequential flows, use the >> operator:
    pipeline = worker1 >> worker2 >> worker3

Example - Self-correcting RAG with cycle:
    graph = Graph()
    graph.add("retrieve", retrieve_worker)
    graph.add("critique", critique_worker)
    graph.add("refine", refine_worker)
    graph.add("generate", generate_worker)

    graph.set_entry("retrieve")
    graph.edge("retrieve", "critique")
    graph.edge("critique", lambda ctx: "refine" if ctx.quality < 0.8 else "generate")
    graph.edge("refine", "retrieve")  # Cycle!
    graph.edge("generate", END)

    compiled = graph.compile()
    result = await compiled.arun(ctx)

Example - Fan-out/fan-in parallel execution:
    graph = Graph()
    graph.add("query", query_worker)
    graph.add("vector_search", vector_worker)
    graph.add("web_search", web_worker)
    graph.add("rerank", rerank_worker)

    graph.set_entry("query")
    # Fan-out: query feeds both searches in parallel
    graph.edge("query", ["vector_search", "web_search"])
    # Fan-in: both searches feed into rerank
    graph.edge("vector_search", "rerank")
    graph.edge("web_search", "rerank")
    graph.edge("rerank", END)

    compiled = graph.compile()
    result = await compiled.arun(ctx)
"""

import warnings
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional, Set

from .constants import END
from .context import Context

if TYPE_CHECKING:
    from .compiled_graph import CompiledGraph

# Edge target can be:
# - str: single target node
# - list[str]: fan-out to multiple targets
# - Callable[[Context], str | list[str]]: dynamic routing
EdgeTarget = str | list[str] | Callable[[Context], str | list[str]]


class Graph:
    """
    Async-first graph supporting arbitrary cycles, conditional routing, and parallel execution.

    Graphs can contain:
    - Workers (leaf computations)
    - Other Graphs (subgraphs)
    - Pipelines (sequential graphs)
    - Any Executable

    All composition is natural - just add executables as nodes.
    Fan-out and fan-in are automatic based on edge topology.

    Graphs must be compiled before execution:
        graph = Graph()
        graph.add("a", worker_a)
        graph.add("b", worker_b)
        graph.edge("a", "b")

        compiled = graph.compile()
        result = await compiled.arun(ctx)
    """

    def __init__(self, name: str = "graph", max_steps: Optional[int] = None):
        """
        Initialize a new graph.

        Args:
            name: Name for this graph (used in logging)
            max_steps: Optional maximum execution steps to prevent infinite loops.
                      If None (default), no limit is enforced.
        """
        self.name = name
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, EdgeTarget] = {}
        self.entry_point: Optional[str] = None
        self.max_steps = max_steps

    def add(self, name: str, executable: Any) -> "Graph":
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
            graph.add("worker", MyWorker())
            graph.add("subgraph", another_graph)
            graph.add("pipeline", worker1 >> worker2)
            graph.add("custom", lambda ctx: ctx)
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")

        self.nodes[name] = executable

        # Auto-set entry point if this is the first node
        if self.entry_point is None:
            self.entry_point = name

        return self

    def edge(self, source: str, target: EdgeTarget) -> "Graph":
        """
        Add an edge from source node to target(s).

        The target can be:
        - str: Single target node (or END to terminate)
        - list[str]: Fan-out to multiple targets in parallel
        - Callable[[Context], str | list[str]]: Dynamic routing based on context

        Args:
            source: Source node name
            target: Target node(s) or routing function

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source node doesn't exist, or static targets don't exist

        Examples:
            # Static edge
            graph.edge("a", "b")

            # Fan-out (parallel execution)
            graph.edge("query", ["vector_search", "web_search"])

            # Conditional routing (named function)
            def route_by_quality(ctx: Context) -> str:
                return "generate" if ctx.quality > 0.8 else "refine"
            graph.edge("critique", route_by_quality)

            # Conditional routing (lambda)
            graph.edge("critique", lambda ctx: "generate" if ctx.quality > 0.8 else "refine")

            # Conditional fan-out
            graph.edge("classify", lambda ctx: ["email", "sms"] if ctx.urgent else ["email"])
        """
        self._ensure_node_exists(source)

        # Validate static targets (not callables)
        if isinstance(target, str):
            self._assert_valid_target(target)
        elif isinstance(target, list):
            for t in target:
                self._assert_valid_target(t)
        # Callables are validated at runtime

        # If source already has an edge, we need to handle it
        if source in self.edges:
            existing = self.edges[source]
            # If both are static, merge them into a list (fan-out)
            if isinstance(existing, str) and isinstance(target, str):
                self.edges[source] = [existing, target]
            elif isinstance(existing, list) and isinstance(target, str):
                if target not in existing:
                    existing.append(target)
            elif isinstance(existing, list) and isinstance(target, list):
                for t in target:
                    if t not in existing:
                        existing.append(t)
            else:
                # One is callable - can't merge, replace
                raise ValueError(
                    f"Node '{source}' already has an edge. "
                    f"Cannot mix callable routing with multiple edge() calls. "
                    f"Use a single edge() call with all targets."
                )
        else:
            self.edges[source] = target

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

    def compile(self, *, flatten: bool = True) -> "CompiledGraph":
        """
        Compile graph into an executable representation.

        This validates the graph structure and returns an immutable
        executor. Build graphs once, compile, and then execute the
        compiled artifact many times.

        Args:
            flatten: If True (default), inline all subgraphs into a flat structure.
                    If False, preserve subgraph boundaries (nested execution).

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValueError: If graph structure is invalid

        Example:
            graph = Graph()
            graph.add("a", worker_a)
            graph.add("b", subgraph_b)
            graph.edge("a", "b")

            # Flattened (default) - all nodes visible
            flat = graph.compile()
            result = await flat.arun(ctx)

            # Nested - subgraphs execute as units
            nested = graph.compile(flatten=False)
            result = await nested.arun(ctx)
        """
        self._validate()

        if flatten:
            return self._compile_flat()
        else:
            return self._compile_nested()

    def _compile_flat(self) -> "CompiledGraph":
        """Compile with all subgraphs recursively inlined."""
        from .compiled_graph import CompiledGraph
        from .graph_flattener import GraphFlattener

        flattener = GraphFlattener(self)
        flat_nodes, flat_edges, new_entry = flattener.flatten()

        if new_entry is None:
            raise ValueError("Entry point missing when flattening graph")

        return CompiledGraph(
            name=f"{self.name}_compiled",
            nodes=flat_nodes,
            edges=flat_edges,
            entry_point=new_entry,
            max_steps=self.max_steps,
        )

    def _compile_nested(self) -> "CompiledGraph":
        """Compile preserving subgraph boundaries."""
        from .compiled_graph import CompiledGraph

        if self.entry_point is None:
            raise ValueError("Entry point not set")

        # Recursively compile any subgraphs, but don't flatten
        compiled_nodes: dict[str, Any] = {}
        for name, node in self.nodes.items():
            if isinstance(node, Graph):
                # Compile subgraph (also nested)
                compiled_nodes[name] = node.compile(flatten=False)
            else:
                compiled_nodes[name] = node

        return CompiledGraph(
            name=f"{self.name}_compiled",
            nodes=compiled_nodes,
            edges=self.edges.copy(),
            entry_point=self.entry_point,
            max_steps=self.max_steps,
        )

    def _validate(self) -> None:
        """Validate graph structure."""
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
        dead_ends = [
            node for node in self.nodes if node not in self.edges
        ]
        if dead_ends:
            warnings.warn(
                f"Nodes without outgoing edges: {dead_ends}. "
                f"Consider adding edges to END."
            )

    def _find_reachable_nodes(self) -> Set[str]:
        """BFS to find all reachable nodes from entry point."""
        reachable: Set[str] = set()
        queue = deque([self.entry_point])

        while queue:
            current = queue.popleft()
            if current in reachable or current == END:
                continue

            reachable.add(current)

            edge_value = self.edges.get(current)
            if edge_value is None:
                continue

            targets = self._get_static_targets(edge_value)
            for target in targets:
                if target != END:
                    queue.append(target)

        return reachable

    def _get_static_targets(self, edge: EdgeTarget) -> list[str]:
        """Extract statically known targets from an edge (for validation)."""
        if isinstance(edge, str):
            return [edge]
        elif isinstance(edge, list):
            return edge
        else:
            # Callable - can't know targets statically, return empty
            # This means we can't detect unreachable nodes behind conditionals
            return []

    def visualize(self) -> str:
        """
        Generate Mermaid diagram syntax for the graph.

        Returns:
            Mermaid diagram code
        """
        from .visualization import generate_mermaid_diagram

        if self.entry_point is None:
            raise ValueError("Entry point not set")

        return generate_mermaid_diagram(self.nodes, self.edges, self.entry_point)

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, entry='{self.entry_point}')"

    def _ensure_node_exists(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

    def _assert_valid_target(self, target: str) -> None:
        if target == END:
            return
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' does not exist")
