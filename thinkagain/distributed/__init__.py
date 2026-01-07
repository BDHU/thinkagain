"""Distributed execution components."""

from __future__ import annotations

from .autoscaling import AutoScaler
from .manager import replica
from .nodes import NodeConfig
from .optimizer import (
    Constraints,
    DeploymentPlan,
    compare_scenarios,
    optimize,
)
from .profiling import profile
from .runtime import init


# Backward compatibility - keep these but not in primary docs
from .manager import (  # noqa: F401
    ReplicaManager,
    get_default_manager,
    set_default_manager,
)
from .replica import ReplicaSpec  # noqa: F401
from .runtime import (  # noqa: F401
    get_runtime_config,
    list_backends,
    register_backend,
    reset_backend,
    runtime,
)


__all__ = [
    # Core API (primary interface)
    "init",
    "replica",
    "optimize",
    "profile",
    # Scaling
    "AutoScaler",
    # Configuration
    "Constraints",
    "NodeConfig",
    # Results
    "DeploymentPlan",
    # Utilities
    "compare_scenarios",
]
