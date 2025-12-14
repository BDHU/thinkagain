"""Immutable executable graph, created by Graph.compile()."""

from typing import Any, AsyncIterator, Dict, Optional

from .constants import END
from .context import Context
from .executable import Executable
from .runtime import EdgeMap, StreamEvent, execute_graph, stream_graph_events


class CompiledGraph(Executable):
    """Executable graph. Created by Graph.compile(), not instantiated directly."""

    def __init__(
        self,
        name: str,
        nodes: Dict[str, Any],
        edges: EdgeMap,
        entry_point: str,
        max_steps: Optional[int] = None,
    ):
        super().__init__(name)
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.max_steps = max_steps

    async def arun(self, ctx: Context) -> Context:
        return await execute_graph(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=f"[{self.name}]",
        )

    async def stream(self, ctx: Context) -> AsyncIterator[StreamEvent]:
        async for event in stream_graph_events(
            ctx=ctx,
            nodes=self.nodes,
            edges=self.edges,
            entry_point=self.entry_point,
            max_steps=self.max_steps,
            end_token=END,
            log_prefix=f"[{self.name}]",
        ):
            yield event

    def visualize(self) -> str:
        """Generate Mermaid diagram syntax."""
        return self._generate_mermaid(self.nodes, self.edges, self.entry_point)

    @staticmethod
    def _generate_mermaid(
        nodes: Dict[str, Any],
        edges: EdgeMap,
        entry_point: str,
    ) -> str:
        from .graph import Graph

        lines = ["graph TD", f"    START([START]) --> {entry_point}"]

        for node_name, node in nodes.items():
            if isinstance(node, (Graph, CompiledGraph)):
                label = f"{node_name}\\n(subgraph: {node.name})"
                lines.append(f'    {node_name}[["{label}"]]')
            else:
                lines.append(f"    {node_name}[{node_name}]")

        lines.append("    END([END])")

        for from_node, edge_value in edges.items():
            if isinstance(edge_value, str):
                target = "END" if edge_value == END else edge_value
                lines.append(f"    {from_node} --> {target}")
            elif isinstance(edge_value, list):
                for to_node in edge_value:
                    target = "END" if to_node == END else to_node
                    lines.append(f"    {from_node} --> {target}")
            else:
                lines.append(f"    {from_node} -->|dynamic| ???")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CompiledGraph(name='{self.name}', nodes={len(self.nodes)})"
