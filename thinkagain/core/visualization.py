"""
Shared visualization helpers for graphs.
"""

from typing import Any, Dict

from .constants import END
from .runtime import EdgeTarget, DirectEdge, ConditionalEdge


def generate_mermaid_diagram(
    nodes: Dict[str, Any],
    edges: Dict[str, EdgeTarget],
    entry_point: str,
) -> str:
    """
    Generate Mermaid diagram syntax for a graph structure.

    Args:
        nodes: Mapping of node names to executables
        edges: Mapping of node names to edge targets
        entry_point: Starting node name

    Returns:
        Mermaid diagram code that can be rendered or saved
    """
    # Delay import to avoid circular dependency
    from .graph import Graph
    from .compiled_graph import CompiledGraph

    lines = ["graph TD", f"    START([START]) --> {entry_point}"]

    for node_name, node in nodes.items():
        if isinstance(node, (Graph, CompiledGraph)):
            label = f"{node_name}\\n(subgraph: {node.name})"
            lines.append(f'    {node_name}[["{label}"]]')
        else:
            lines.append(f"    {node_name}[{node_name}]")

    lines.append("    END([END])")

    for from_node, edge in edges.items():
        if isinstance(edge, ConditionalEdge):
            for label, to_node in edge.paths.items():
                target = "END" if to_node == END else to_node
                lines.append(f"    {from_node} -->|{label}| {target}")
        elif isinstance(edge, DirectEdge):
            target = "END" if edge.target == END else edge.target
            lines.append(f"    {from_node} --> {target}")
        else:
            raise TypeError(f"Unknown edge type: {type(edge)}")

    return "\n".join(lines)


__all__ = ["generate_mermaid_diagram"]
