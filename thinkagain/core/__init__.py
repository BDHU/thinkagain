"""Core components - JAX-style graph building API."""

from .devices import CpuDevice, Device, GpuDevice, devices
from .errors import NodeExecutionError, TracingError
from .graph import Graph
from .hooks import ExecutionHook, register_hook, unregister_hook
from .ops import cond, scan, switch, while_loop
from .replica import ReplicaHandle, replica
from .tracing import clear_compiled_cache, get_cache_info, jit, node

__all__ = [
    # Decorators
    "jit",
    "node",
    "replica",
    # Control flow
    "cond",
    "while_loop",
    "scan",
    "switch",
    # Types
    "Graph",
    "ReplicaHandle",
    # Devices
    "Device",
    "GpuDevice",
    "CpuDevice",
    "devices",
    # Errors
    "NodeExecutionError",
    "TracingError",
    # Utilities
    "clear_compiled_cache",
    "get_cache_info",
    # Hooks (for extensions)
    "ExecutionHook",
    "register_hook",
    "unregister_hook",
]
