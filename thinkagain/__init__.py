"""thinkagain - Minimal framework for declarative AI pipelines with JAX-style graph building."""

from .core import (
    Bundle,
    CpuDevice,
    GpuDevice,
    Graph,
    NodeExecutionError,
    ReplicaHandle,
    TracingError,
    apply_replica,
    bind_service,
    bundle,
    cond,
    devices,
    jit,
    make_inputs,
    node,
    replica,
    scan,
    switch,
    trace,
    while_loop,
)
from .core.profiling import (
    disable_profiling,
    enable_profiling,
    get_profiler,
    is_profiling_enabled,
    profile,
)
from .distributed import (
    Mesh,
    MeshNode,
    get_current_mesh,
    register_distributed_hooks,
    unregister_distributed_hooks,
)

__version__ = "0.3.0"

__all__ = [
    # Core - Node decorator
    "node",
    "bind_service",
    # Core - Replica decorator
    "replica",
    "apply_replica",
    # Core - Traceable types
    "trace",
    # Core - Compilation
    "jit",
    "Graph",
    # Core - Control flow
    "cond",
    "while_loop",
    "scan",
    "switch",
    # Core - Types
    "ReplicaHandle",
    # Core - Input bundling
    "Bundle",
    "bundle",
    "subset",
    "extend",
    "replace",
    "get",
    "unpack",
    "make_inputs",
    # Core - Errors
    "NodeExecutionError",
    "TracingError",
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

# Top-level Bundle ops for user-facing API.
subset = bundle.subset
extend = bundle.extend
replace = bundle.replace
get = bundle.get
unpack = bundle.unpack
