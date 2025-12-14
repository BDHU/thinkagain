"""
Graph flattening logic for inlining nested subgraphs.

This module provides the GraphFlattener helper that recursively expands
all Graph nodes into a flat structure with prefixed node names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .constants import END
from .runtime import EdgeMap, EdgeTarget

if TYPE_CHECKING:
    from .graph import Graph


class GraphFlattener:
    """
    Helper that rewrites a nested Graph into a flat node/edge map.

    Breaking the flattening logic into a dedicated helper keeps the main
    Graph class easier to scan while providing a single place to reason
    about recursion, prefix naming, and cycle detection.
    """

    def __init__(self, root: "Graph"):
        self.root = root
        self.flat_nodes: Dict[str, Any] = {}
        self.flat_edges: EdgeMap = {}
        self._visited: Set[int] = set()

    def flatten(self) -> Tuple[Dict[str, Any], EdgeMap, Optional[str]]:
        """
        Flatten the graph, inlining all subgraphs.

        Returns:
            Tuple of (flat_nodes, flat_edges, entry_point)
        """
        node_mapping = {
            node_name: self._flatten_node(node_name, node)
            for node_name, node in self.root.nodes.items()
        }

        for from_node, edge_value in self.root.edges.items():
            _, from_exit = node_mapping[from_node]
            self.flat_edges[from_exit] = self._rewrite_edge(edge_value, node_mapping, END)

        entry = None
        if self.root.entry_point is not None:
            entry, _ = node_mapping[self.root.entry_point]

        return self.flat_nodes, self.flat_edges, entry

    def _flatten_node(
        self, node_name: str, node: Any, prefix: str = ""
    ) -> Tuple[str, str]:
        """
        Flatten a single node.

        Returns:
            Tuple of (entry_name, exit_name) for this node.
            For leaf nodes, both are the same.
            For subgraphs, entry is the subgraph's entry and exit is a virtual END node.
        """
        full_name = f"{prefix}{node_name}" if prefix else node_name

        # Delay import to avoid circular dependency
        from .graph import Graph

        if isinstance(node, Graph):
            return self._flatten_subgraph(full_name, node)

        self.flat_nodes[full_name] = node
        return full_name, full_name

    def _flatten_subgraph(self, full_name: str, graph: "Graph") -> Tuple[str, str]:
        """
        Flatten a subgraph, inlining all its nodes with prefixed names.

        Returns:
            Tuple of (entry_name, virtual_end_name)
        """
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

            for from_node, edge_value in graph.edges.items():
                _, from_exit = sub_mapping[from_node]
                self.flat_edges[from_exit] = self._rewrite_edge(
                    edge_value, sub_mapping, virtual_end
                )

            entry, _ = sub_mapping[graph.entry_point]
            return entry, virtual_end
        finally:
            self._visited.remove(graph_id)

    def _rewrite_edge(
        self,
        edge_value: EdgeTarget,
        mapping: Dict[str, Tuple[str, str]],
        default_target: str,
    ) -> EdgeTarget:
        """
        Rewrite an edge value using the node mapping.

        Args:
            edge_value: The original edge target
            mapping: Maps original node names to (entry, exit) tuples
            default_target: Target to use for END (either global END or virtual END)

        Returns:
            Rewritten edge target
        """
        if isinstance(edge_value, str):
            # Single target
            if edge_value == END:
                return default_target
            return mapping[edge_value][0]  # Use entry point of target

        elif isinstance(edge_value, list):
            # List of targets (fan-out)
            result: List[str] = []
            for target in edge_value:
                if target == END:
                    result.append(default_target)
                else:
                    result.append(mapping[target][0])
            return result

        else:
            # Callable - wrap it to rewrite its return values
            original_fn = edge_value

            def rewritten_route(ctx: Any) -> Union[str, List[str]]:
                result = original_fn(ctx)
                if isinstance(result, str):
                    if result == END:
                        return default_target
                    return mapping[result][0]
                else:
                    # List of targets
                    return [
                        default_target if t == END else mapping[t][0]
                        for t in result
                    ]

            return rewritten_route


__all__ = ["GraphFlattener"]
