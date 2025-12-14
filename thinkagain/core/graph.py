"""
Graph builder for complex workflows with cycles, conditionals, and parallel execution.

Example:
    graph = Graph()
    graph.add("retrieve", retriever)
    graph.add("generate", generator)
    graph.edge("retrieve", "generate")
    graph.edge("generate", END)
    result = await graph.compile().arun(ctx)
"""

import warnings
from collections import deque
from typing import TYPE_CHECKING, Any, Optional, Set, cast

from .constants import END
from .runtime import EdgeMap, EdgeTarget

if TYPE_CHECKING:
    from .compiled_graph import CompiledGraph


class Graph:
    """
    Builder for executable graphs. Call compile() to get an executable.

    Supports cycles, conditional routing, fan-out/fan-in, and nested subgraphs.
    """

    def __init__(self, name: str = "graph", max_steps: Optional[int] = None):
        self.name = name
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, EdgeTarget] = {}
        self.entry_point: Optional[str] = None
        self.max_steps = max_steps

    def add(self, name: str, executable: Any) -> "Graph":
        """Add a node. First node becomes entry point."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = executable
        if self.entry_point is None:
            self.entry_point = name
        return self

    def edge(self, source: str, target: EdgeTarget) -> "Graph":
        """
        Add edge from source to target(s).

        Target can be: str, list[str], or Callable[[Context], str | list[str]].
        Multiple calls from same source merge into fan-out (for static targets).
        """
        self._ensure_node_exists(source)
        self._validate_targets(target)

        if source not in self.edges:
            self.edges[source] = target
            return self

        existing = self.edges[source]

        if isinstance(existing, list):
            existing_list = list(existing)
        elif isinstance(existing, str):
            existing_list = [existing]
        else:
            raise ValueError(
                f"Node '{source}' already has an edge. Cannot mix callable routing."
            )

        if isinstance(target, list):
            new_targets = list(target)
        elif isinstance(target, str):
            new_targets = [target]
        else:
            raise ValueError(
                f"Node '{source}' already has an edge. Cannot mix callable routing."
            )

        for t in new_targets:
            if t not in existing_list:
                existing_list.append(t)

        self.edges[source] = (
            existing_list if len(existing_list) > 1 else existing_list[0]
        )
        return self

    def set_entry(self, name: str) -> "Graph":
        """Set the starting node."""
        self._ensure_node_exists(name)
        self.entry_point = name
        return self

    def compile(self, *, flatten: bool = True) -> "CompiledGraph":
        """Compile into executable. Validates structure."""
        self._validate()

        if flatten:
            nodes, edges, entry = self._flatten()
        else:
            nodes, edges, entry = self._compile_nested()

        from .compiled_graph import CompiledGraph

        return CompiledGraph(
            name=f"{self.name}_compiled",
            nodes=nodes,
            edges=edges,
            entry_point=entry,
            max_steps=self.max_steps,
        )

    def _flatten(self) -> tuple[dict[str, Any], EdgeMap, str]:
        from .graph_flattener import GraphFlattener

        if self.entry_point is None:
            raise ValueError("Entry point not set")

        return GraphFlattener(
            nodes=self.nodes,
            edges=self._snapshot_edges(),
            entry=self.entry_point,
        ).flatten()

    def _compile_nested(self) -> tuple[dict[str, Any], EdgeMap, str]:
        if self.entry_point is None:
            raise ValueError("Entry point not set")

        compiled_nodes: dict[str, Any] = {}
        for name, node in self.nodes.items():
            if isinstance(node, Graph):
                compiled_nodes[name] = node.compile(flatten=False)
            else:
                compiled_nodes[name] = node

        return compiled_nodes, self._snapshot_edges(), self.entry_point

    def _snapshot_edges(self) -> EdgeMap:
        snapshot: EdgeMap = {}
        for source, target in self.edges.items():
            if isinstance(target, list):
                snapshot[source] = list(target)
            else:
                snapshot[source] = target
        return snapshot

    def _validate(self) -> None:
        if self.entry_point is None:
            raise ValueError(
                "Entry point not set. Use set_entry() or add the first node."
            )

        reachable = self._find_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            warnings.warn(f"Unreachable nodes: {unreachable}")

        dead_ends = [n for n in self.nodes if n not in self.edges]
        if dead_ends:
            warnings.warn(
                f"Nodes without outgoing edges: {dead_ends}. Add edges to END."
            )

    def _find_reachable_nodes(self) -> Set[str]:
        if self.entry_point is None:
            return set()

        reachable: Set[str] = set()
        queue: deque[str] = deque([self.entry_point])

        while queue:
            current = queue.popleft()
            if current in reachable or current == END:
                continue
            reachable.add(current)

            edge = self.edges.get(current)
            if edge is None:
                continue

            for t in self._get_static_targets(edge):
                if t != END:
                    queue.append(t)

        return reachable

    def _get_static_targets(self, edge: EdgeTarget) -> list[str]:
        if isinstance(edge, str):
            return [edge]
        if isinstance(edge, list):
            return list(cast(list[str], edge))
        return []

    def visualize(self) -> str:
        """Generate Mermaid diagram syntax."""
        from .compiled_graph import CompiledGraph

        if self.entry_point is None:
            raise ValueError("Entry point not set")

        return CompiledGraph._generate_mermaid(self.nodes, self.edges, self.entry_point)

    def _ensure_node_exists(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

    def _validate_targets(self, target: EdgeTarget) -> None:
        if isinstance(target, str):
            if target != END and target not in self.nodes:
                raise ValueError(f"Target node '{target}' does not exist")
        elif isinstance(target, list):
            for t in target:
                if isinstance(t, str) and t != END and t not in self.nodes:
                    raise ValueError(f"Target node '{t}' does not exist")

    def __repr__(self) -> str:
        return f"Graph(name='{self.name}', nodes={len(self.nodes)}, entry='{self.entry_point}')"
