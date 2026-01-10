"""Distributed execution components."""

from __future__ import annotations

from .mesh import Mesh, MeshNode, get_current_mesh

__all__ = [
    "Mesh",
    "MeshNode",
    "get_current_mesh",
]

# Auto-register distributed execution hook when this module is imported
# This allows core executor to handle @replica classes without direct coupling
from ..core.execution.hooks import register_hook
from .execution_hook import distributed_execution_hook

register_hook(distributed_execution_hook)
