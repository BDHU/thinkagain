"""
Core components of the minimal agent framework.

Everything is built on a simple hierarchy:
- Executable: Base interface for all components
- Worker: Leaf computations (your business logic)
- Graph: DAG with cycles and conditional routing
- CompiledGraph: Immutable executable representation

All components compose with >> operator.
Use graph.compile() to separate building from execution.
"""

from .constants import END
from .context import Context
from .executable import Executable
from .worker import Worker
from .graph import Graph
from .compiled_graph import CompiledGraph
from .runtime import StreamEvent, DirectEdge, ConditionalEdge

__all__ = [
    "Context",
    "Executable",
    "Worker",
    "Graph",
    "END",
    "CompiledGraph",
    "StreamEvent",
    "DirectEdge",
    "ConditionalEdge",
]
