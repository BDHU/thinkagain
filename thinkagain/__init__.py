"""thinkagain - Minimal framework for declarative AI pipelines."""

from contextlib import contextmanager

from .core import Context, Node, node, run, arun, NodeExecutionError, NodeSignatureError
from .distributed import worker, launch, shutdown, WorkerServiceError

__version__ = "0.2.0"


@contextmanager
def runtime():
    """Context manager for worker lifecycle.

    Usage:
        with thinkagain.runtime():
            result = run(pipeline, data)
    """
    launch()
    try:
        yield
    finally:
        shutdown()


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
    "worker",
    "launch",
    "shutdown",
    "runtime",
    "WorkerServiceError",
]
