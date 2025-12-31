"""Distributed execution components."""

from __future__ import annotations

from .manager import (
    ReplicaManager,
    get_default_manager,
    replica,
    set_default_manager,
)
from .replica import ReplicaSpec
from .runtime import (
    get_runtime_config,
    init,
    list_backends,
    register_backend,
    reset_backend,
    runtime,
)


__all__ = [
    "ReplicaManager",
    "ReplicaSpec",
    "replica",
    "get_default_manager",
    "set_default_manager",
    "init",
    "get_runtime_config",
    "list_backends",
    "register_backend",
    "reset_backend",
    "runtime",
]
