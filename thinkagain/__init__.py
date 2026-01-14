"""thinkagain - Minimal framework for declarative AI pipelines with dynamic execution."""

from .api import ActorHandle, ReplicaHandle, node, replica
from .resources import CpuDevice, GpuDevice, Mesh, MeshNode, devices, get_current_mesh
from .runtime import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    profile,
    register_distributed_hooks,
    unregister_distributed_hooks,
)
from .runtime.errors import NodeExecutionError

__version__ = "0.3.0"

__all__ = [
    # Core - Node decorator
    "node",
    # Core - Replica decorator
    "replica",
    # Core - Types
    "ReplicaHandle",
    "ActorHandle",
    # Core - Errors
    "NodeExecutionError",
    # Core - Devices
    "GpuDevice",
    "CpuDevice",
    "devices",
    # Distributed - Mesh
    "Mesh",
    "MeshNode",
    "get_current_mesh",
    "register_distributed_hooks",
    "unregister_distributed_hooks",
    # Profiling
    "enable_profiling",
    "disable_profiling",
    "is_profiling_enabled",
    "get_profiler",
    "profile",
]
