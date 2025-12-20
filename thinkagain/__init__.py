"""thinkagain - Minimal framework for declarative AI pipelines."""

from .core.context import Context
from .core.errors import NodeDataclassError, NodeExecutionError, NodeSignatureError
from .core.metadata import ExecutionMetadata
from .core.node import FunctionNode, Node, NodeBase, node
from .core.runner import arun, run
from .distributed import deploy, replica, shutdown

__version__ = "0.2.0"

__all__ = [
    # Core
    "Context",
    "ExecutionMetadata",
    "FunctionNode",
    "Node",
    "NodeBase",
    "NodeDataclassError",
    "NodeExecutionError",
    "NodeSignatureError",
    "arun",
    "node",
    "run",
    # Distributed
    "deploy",
    "replica",
    "shutdown",
]
