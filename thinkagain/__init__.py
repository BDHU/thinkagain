"""thinkagain - Minimal framework for declarative AI pipelines with JAX-style graph building."""

from .core import (
    Graph,
    Node,
    NodeExecutionError,
    cond,
    jit,
    node,
    scan,
    switch,
    while_loop,
)
from .distributed import ReplicaManager, ReplicaSpec, replica

__version__ = "0.3.0"

__all__ = [
    # Core - Node decorator and class
    "node",
    "Node",
    # Core - Compilation
    "jit",
    "Graph",
    # Core - Control flow
    "cond",
    "while_loop",
    "scan",
    "switch",
    # Core - Errors
    "NodeExecutionError",
    # Distributed (legacy - will be updated to use new API)
    "ReplicaManager",
    "ReplicaSpec",
    "replica",
]
