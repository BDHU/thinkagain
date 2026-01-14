"""thinkagain - Minimal framework for declarative AI pipelines with dynamic execution."""

from .api import (
    ResourceConfig,
    AutoscalingConfig,
    ServiceClass,
    ServiceHandle,
    op,
    service,
)
from .resources import (
    CpuDevice,
    GpuDevice,
    Mesh,
    MeshNode,
    devices,
    get_current_mesh,
    require_mesh,
)
from .runtime import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    profile,
    register_distributed_hooks,
    unregister_distributed_hooks,
)
from .runtime.errors import OpExecutionError
from .session import Session

__version__ = "0.3.0"

__all__ = [
    # Core - Op decorator
    "op",
    # Core - Service decorator
    "service",
    # Core - Service config types
    "ResourceConfig",
    "AutoscalingConfig",
    # Core - Types
    "ServiceHandle",
    "ServiceClass",
    # Core - Errors
    "OpExecutionError",
    # Core - Devices
    "GpuDevice",
    "CpuDevice",
    "devices",
    # Distributed - Mesh
    "Mesh",
    "MeshNode",
    "get_current_mesh",
    "require_mesh",
    "register_distributed_hooks",
    "unregister_distributed_hooks",
    # Session - Execution + Optimization
    "Session",
    # Profiling
    "enable_profiling",
    "disable_profiling",
    "is_profiling_enabled",
    "get_profiler",
    "profile",
]
