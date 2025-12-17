"""thinkagain - Minimal framework for declarative AI pipelines."""

from .core import Context, Node, node, run, arun, NodeExecutionError, NodeSignatureError
from .distributed import replica, deploy, shutdown

__version__ = "0.2.0"

__all__ = [
    # Core
    "Context",
    "Node",
    "node",
    "run",
    "arun",
    "NodeExecutionError",
    "NodeSignatureError",
    # Distributed
    "replica",
    "deploy",
    "shutdown",
]
