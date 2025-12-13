"""
Graph flattening logic for inlining nested subgraphs.

This module provides the _GraphFlattener helper that recursively expands
all Graph nodes into a flat structure with prefixed node names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

from .constants import END
from .runtime import EdgeTarget, DirectEdge, ConditionalEdge

if TYPE_CHECKING:
    from .graph import Graph


class GraphFlattener:
    """
    Helper that rewrites a nested Graph into a flat node/edge map.

    Breaking the flattening logic into a dedicated helper keeps the main
    Graph class easier to scan while providing a single place to reason
    about recursion, prefix naming, and cycle detection.
    """

    def __init__(self, root: Graph):
        self.root = root
        self.flat_nodes: Dict[str, Any] = {}
        self.flat_edges: Dict[str, EdgeTarget] = {}
        self._visited: Set[int] = set()

    def flatten(self) -> Tuple[Dict[str, Any], Dict[str, EdgeTarget], Optional[str]]:
        node_mapping = {
            node_name: self._flatten_node(node_name, node)
            for node_name, node in self.root.nodes.items()
        }

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

        # Delay import to avoid circular dependency
        from .graph import Graph

        if isinstance(node, Graph):
            return self._flatten_subgraph(full_name, node)

        self.flat_nodes[full_name] = node
        return full_name, full_name

    def _flatten_subgraph(self, full_name: str, graph: Graph) -> Tuple[str, str]:
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
        if isinstance(edge, ConditionalEdge):
            updated_paths: Dict[str, str] = {}
            for label, target in edge.paths.items():
                if target == END:
                    updated_paths[label] = default_target
                else:
                    updated_paths[label] = mapping[target][0]
            return ConditionalEdge(route_fn=edge.route_fn, paths=updated_paths)

        if isinstance(edge, DirectEdge):
            if edge.target == END:
                return DirectEdge(target=default_target)
            return DirectEdge(target=mapping[edge.target][0])

        # Should not reach here with new edge classes
        raise TypeError(f"Unknown edge type: {type(edge)}")


__all__ = ["GraphFlattener"]
