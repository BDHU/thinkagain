"""
Core components of the minimal agent framework.

Everything is built on a simple hierarchy:
- Executable: Base interface for all components
- Worker: Leaf computations (your business logic)
- Graph: DAG with cycles and conditional routing
- Pipeline: Sequential graphs (syntactic sugar)
- CompiledGraph: Immutable executable representation

All components compose with >> operator.
Use graph.compile() to separate building from execution.
"""

from .context import Context
from .executable import Executable
from .worker import Worker
from .graph import Graph, END
from .pipeline import Pipeline
from .compiled_graph import CompiledGraph

__all__ = [
    "Context",
    "Executable",
    "Worker",
    "Graph",
    "END",
    "Pipeline",
    "CompiledGraph",
]
