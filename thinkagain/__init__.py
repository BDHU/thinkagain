"""
thinkagain - A minimal agent framework where everything is a Graph.

Core concepts:
- Context: Shared state that flows through execution
- Executable: Base interface for all components
- Worker: Leaf computations (your business logic)
- Graph: DAG with cycles and conditional routing

Everything composes with >> operator.
Graphs can contain graphs (subgraphs) naturally.
"""

from .core import Context, Executable, Worker, Graph, END, StreamEvent
from .core.worker import async_worker

__version__ = "0.1.1"

__all__ = [
    "Context",
    "Executable",
    "Worker",
    "Graph",
    "END",
    "StreamEvent",
    "async_worker",
]
