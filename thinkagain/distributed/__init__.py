"""Distributed execution components."""

from __future__ import annotations

from .mesh import Mesh, MeshNode, get_current_mesh

__all__ = [
    "Mesh",
    "MeshNode",
    "get_current_mesh",
    "register_distributed_hooks",
    "unregister_distributed_hooks",
]


def register_distributed_hooks() -> None:
    """Register distributed execution hooks explicitly."""
    from ..core.execution.hooks import register_hook
    from .execution_hook import distributed_execution_hook

    register_hook(distributed_execution_hook)


def unregister_distributed_hooks() -> None:
    """Unregister distributed execution hooks explicitly."""
    from ..core.execution.hooks import unregister_hook
    from .execution_hook import distributed_execution_hook

    unregister_hook(distributed_execution_hook)
