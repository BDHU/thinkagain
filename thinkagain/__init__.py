"""thinkagain - Minimal framework for declarative AI pipelines with JAX-style graph building."""

from .core import (
    CpuDevice,
    GpuDevice,
    Graph,
    NodeExecutionError,
    cond,
    devices,
    jit,
    node,
    scan,
    switch,
    while_loop,
)
from .core.profiling import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    profile,
)
from .distributed import Mesh, MeshNode, get_current_mesh
from .distributed.replication import replicate

__version__ = "0.3.0"

__all__ = [
    # Core - Node decorator
    "node",
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
    # Core - Devices
    "GpuDevice",
    "CpuDevice",
    "devices",
    # Distributed - Mesh
    "Mesh",
    "MeshNode",
    "get_current_mesh",
    # Distributed - Replication
    "replicate",
    # Profiling
    "enable_profiling",
    "disable_profiling",
    "is_profiling_enabled",
    "get_profiler",
    "profile",
]
