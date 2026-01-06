"""thinkagain - Minimal framework for declarative AI pipelines."""

from .core.context import Context
from .core.errors import NodeDataclassError, NodeExecutionError
from .core.node import FunctionNode, Node, NodeBase, node
from .core.runner import arun, chain, run
from .core.sequential import Sequential
from .distributed import ReplicaManager, ReplicaSpec, replica

__version__ = "0.2.0"

__all__ = [
    # Core
    "Context",
    "FunctionNode",
    "Node",
    "NodeBase",
    "NodeDataclassError",
    "NodeExecutionError",
    "Sequential",
    "arun",
    "chain",
    "node",
    "run",
    # Distributed
    "ReplicaManager",
    "ReplicaSpec",
    "replica",
]
