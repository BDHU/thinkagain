"""
Pipeline - syntactic sugar for sequential graphs.

Pipeline is just a Graph with sequential auto-wiring.
Use >> operator for the most concise sequential composition:

    pipeline = worker1 >> worker2 >> worker3

This is equivalent to:

    graph = Graph()
    graph.add_node("0", worker1)
    graph.add_node("1", worker2)
    graph.add_node("2", worker3)
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")
    graph.add_edge("2", END)

Everything is a graph under the hood.
"""

from typing import List
from .graph import Graph, END
from .executable import Executable


class Pipeline(Graph):
    """
    Sequential graph with auto-wiring.

    Pipeline is syntactic sugar over Graph - it automatically connects
    nodes in sequence. Under the hood, it's just a Graph.

    Prefer the >> operator for concise composition:
        pipeline = worker1 >> worker2 >> worker3

    Or use Pipeline() explicitly:
        pipeline = Pipeline([worker1, worker2, worker3])

    Both create the same underlying Graph structure.
    """

    def __init__(self, nodes: List[Executable] = None, name: str = "pipeline"):
        """
        Initialize pipeline with sequential nodes.

        Args:
            nodes: List of executables to run in sequence
            name: Name for this pipeline

        Example:
            # Explicit pipeline creation
            pipeline = Pipeline([
                VectorDB(),
                Reranker(),
                Generator()
            ], name="rag_pipeline")

            # Equivalent using >> operator
            pipeline = VectorDB() >> Reranker() >> Generator()
        """
        super().__init__(name=name)

        self._build_linear_chain(nodes or [])

    def _build_linear_chain(self, nodes: List[Executable]) -> None:
        """Wire a simple left-to-right graph for the provided nodes."""
        previous = None
        for index, node in enumerate(nodes):
            name = f"_{index}"
            self.add_node(name, node)

            if previous is None:
                self.set_entry(name)
            else:
                self.add_edge(previous, name)

            previous = name

        if previous is not None:
            self.add_edge(previous, END)

    def __repr__(self) -> str:
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)})"
