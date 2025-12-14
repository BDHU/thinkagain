"""
thinkagain - A minimal agent framework where everything is a Graph.

Core concepts:
- Context: Shared state that flows through execution
- Executable: Base class for components (implement arun, optionally astream)
- Graph: Builder for DAGs with cycles and conditional routing

Compose workflows by building Graph instances (add nodes, connect edges,
then compile). Graphs can contain graphs (subgraphs) naturally.
"""

from .core import (
    Context,
    Executable,
    async_executable,
    Graph,
    CompiledGraph,
    END,
    StreamEvent,
)

__version__ = "0.1.1"

__all__ = [
    "Context",
    "Executable",
    "async_executable",
    "Graph",
    "CompiledGraph",
    "END",
    "StreamEvent",
]
