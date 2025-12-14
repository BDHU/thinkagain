"""
Core components of the minimal agent framework.

Everything is built on a simple hierarchy:
- Executable: Base interface for all components
- Worker: Leaf computations (your business logic)
- Graph: Builder for graph structures
- CompiledGraph: Executable graph representation

All components compose with >> operator.
Use graph.compile() to get an executable representation.

Example:
    # Simple pipeline
    pipeline = worker1 >> worker2 >> worker3
    result = await pipeline.compile().arun(ctx)

    # Complex graph with routing
    graph = Graph()
    graph.add("a", worker_a)
    graph.add("b", worker_b)
    graph.edge("a", lambda ctx: "b" if ctx.ready else END)
    graph.edge("b", END)

    result = await graph.compile().arun(ctx)
"""

from .constants import END
from .context import Context
from .executable import Executable
from .worker import Worker
from .graph import Graph
from .compiled_graph import CompiledGraph
from .runtime import StreamEvent

__all__ = [
    "Context",
    "Executable",
    "Worker",
    "Graph",
    "END",
    "CompiledGraph",
    "StreamEvent",
]
