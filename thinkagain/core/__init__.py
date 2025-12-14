"""
Core components for building executable graphs.

- Executable: Base class (implement arun, optionally astream)
- Graph: Builder for graph structures
- CompiledGraph: Executable graph from Graph.compile()

Example:
    graph = Graph()
    graph.add("a", my_executable)
    graph.add("b", another_executable)
    graph.edge("a", "b")
    graph.edge("b", END)
    result = await graph.compile().arun(ctx)
"""

from .constants import END
from .context import Context
from .executable import Executable, async_executable
from .graph import Graph
from .compiled_graph import CompiledGraph
from .runtime import StreamEvent

__all__ = [
    "Context",
    "Executable",
    "async_executable",
    "Graph",
    "END",
    "CompiledGraph",
    "StreamEvent",
]
