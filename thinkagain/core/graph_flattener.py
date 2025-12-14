"""Flattening logic for inlining nested subgraphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Union, cast

from .constants import END
from .runtime import EdgeMap, EdgeTarget

if TYPE_CHECKING:
    from .graph import Graph


class GraphFlattener:
    """Rewrites a nested Graph into a flat node/edge map with prefixed names."""

    def __init__(self, nodes: Dict[str, Any], edges: EdgeMap, entry: str):
        self.nodes = nodes
        self.edges = edges
        self.entry = entry
        self.flat_nodes: Dict[str, Any] = {}
        self.flat_edges: EdgeMap = {}
        self._visited: Set[int] = set()

    def flatten(self) -> Tuple[Dict[str, Any], EdgeMap, str]:
        """Returns (flat_nodes, flat_edges, entry_point)."""
        node_mapping = {
            name: self._flatten_node(name, node) for name, node in self.nodes.items()
        }

        for from_node, edge_value in self.edges.items():
            _, from_exit = node_mapping[from_node]
            self.flat_edges[from_exit] = self._rewrite_edge(
                edge_value, node_mapping, END
            )

        entry, _ = node_mapping[self.entry]
        return self.flat_nodes, self.flat_edges, entry

    def _flatten_node(self, name: str, node: Any, prefix: str = "") -> Tuple[str, str]:
        """Returns (entry_name, exit_name). For leaf nodes, both are the same."""
        full_name = f"{prefix}{name}" if prefix else name

        from .graph import Graph

        if isinstance(node, Graph):
            return self._flatten_subgraph(full_name, node)

        self.flat_nodes[full_name] = node
        return full_name, full_name

    def _flatten_subgraph(self, full_name: str, graph: "Graph") -> Tuple[str, str]:
        """Returns (entry_name, virtual_end_name)."""
        graph_id = id(graph)
        if graph_id in self._visited:
            raise ValueError(
                f"Subgraph cycle detected: '{graph.name}' contains itself."
            )

        self._visited.add(graph_id)
        try:
            if graph.entry_point is None:
                raise ValueError(f"Subgraph '{graph.name}' missing entry point")

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
        """Rewrite edge targets using the node mapping."""
        if isinstance(edge_value, str):
            return default_target if edge_value == END else mapping[edge_value][0]

        if isinstance(edge_value, list):
            return [
                default_target if t == END else mapping[t][0]
                for t in cast(List[str], edge_value)
            ]

        # Callable - wrap to rewrite return values
        original_fn = edge_value

        def rewritten_route(ctx: Any) -> Union[str, List[str]]:
            result = original_fn(ctx)
            if isinstance(result, str):
                return default_target if result == END else mapping[result][0]
            return [default_target if t == END else mapping[t][0] for t in result]

        return rewritten_route


__all__ = ["GraphFlattener"]
