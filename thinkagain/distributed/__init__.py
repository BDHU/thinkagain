"""Distributed execution components."""

from __future__ import annotations

from .mesh import Mesh, MeshNode, get_current_mesh
from .replication import replicate

__all__ = [
    "replicate",
    "Mesh",
    "MeshNode",
    "get_current_mesh",
]

# Auto-register distributed execution hook when this module is imported
# This allows core executor to handle @replicate functions without direct coupling
from ..core.hooks import register_hook
from .execution_hook import distributed_execution_hook

register_hook(distributed_execution_hook)
