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
from .replica import ReplicaConfig, ReplicaHandle, replica
from .replica_state import apply_replica
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
    "apply_replica",
    # Hooks
    "ExecutionHook",
    "register_hook",
    "unregister_hook",
    # Utilities
    "maybe_await",
]
