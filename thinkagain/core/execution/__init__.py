"""Execution infrastructure for running computation graphs.

This module provides the execution engine, node executors, replica handling,
and execution hooks.
"""

from .executor import execute_graph
from .executors import (
    CallExecutor,
    CondExecutor,
    ScanExecutor,
    SwitchExecutor,
    WhileExecutor,
)
from .hooks import ExecutionHook, register_hook, unregister_hook
from .replica import (
    ReplicaConfig,
    ReplicaHandle,
    register_replica_hook,
    replica,
    unregister_replica_hook,
)
from .runtime import maybe_await

__all__ = [
    # Main execution
    "execute_graph",
    # Node executors
    "CallExecutor",
    "CondExecutor",
    "WhileExecutor",
    "ScanExecutor",
    "SwitchExecutor",
    # Replica support
    "replica",
    "ReplicaHandle",
    "ReplicaConfig",
    "register_replica_hook",
    "unregister_replica_hook",
    # Hooks
    "ExecutionHook",
    "register_hook",
    "unregister_hook",
    # Utilities
    "maybe_await",
]
