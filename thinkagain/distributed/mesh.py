"""Mesh configuration for distributed execution.

This module provides mesh configuration for distributed execution across
multiple nodes. Mesh defines available resources and is used as an execution context.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..core.devices import CpuDevice, Device, GpuDevice


# Thread-local context stack for mesh
_mesh_stack = threading.local()


def _get_mesh_stack() -> list["Mesh"]:
    """Get thread-local mesh stack."""
    if not hasattr(_mesh_stack, "stack"):
        _mesh_stack.stack = []
    return _mesh_stack.stack


# ---------------------------------------------------------------------------
# MeshNode
# ---------------------------------------------------------------------------


@dataclass
class MeshNode:
    """Compute node with multiple devices in a distributed cluster."""

    name: str
    gpus: int = 0
    cpus: int = 0
    endpoint: str | None = None

    def expand(self) -> list[Device]:
        """Expand into individual devices."""
        devices = []
        for i in range(self.gpus):
            devices.append(
                GpuDevice(id=i)
            )  # Note: May need host/endpoint tracking in future
        if self.cpus > 0:
            devices.append(
                CpuDevice(id=0)
            )  # Note: May need host/endpoint tracking in future
        return devices

    def __repr__(self):
        return f"MeshNode({self.name}, gpus={self.gpus}, cpus={self.cpus})"


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


class Mesh:
    """Resource mesh defining available compute for distributed execution.

    Mesh is used as a context manager to define the execution environment
    for replicated functions. It specifies available devices (GPUs, CPUs)
    across single or multiple nodes.

    Example:
        # Local execution with auto-detection
        from thinkagain.core.devices import devices
        mesh = Mesh(devices())

        with mesh:
            result = await pipeline("query")

        # Multi-node cluster
        mesh = Mesh([
            MeshNode("server1", gpus=8, endpoint="server1:8000"),
            MeshNode("server2", gpus=8, endpoint="server2:8000"),
        ])

        with mesh:
            result = await pipeline("query")
    """

    def __init__(
        self,
        devices: list[Device | MeshNode],
        backend: str = "local",
    ):
        """Initialize mesh with devices.

        Args:
            devices: List of Device or MeshNode objects
            backend: Backend type ("local" or "grpc")
        """
        self.backend = backend
        self._nodes = []  # Store original MeshNodes for endpoint tracking

        # Expand nodes to devices if needed
        expanded = []
        for d in devices:
            if isinstance(d, MeshNode):
                self._nodes.append(d)
                expanded.extend(d.expand())
            else:
                expanded.append(d)

        self.devices = expanded
        self._gpus = [d for d in expanded if isinstance(d, GpuDevice)]
        self._cpus = [d for d in expanded if isinstance(d, CpuDevice)]

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in mesh."""
        return len(self._gpus)

    @property
    def total_cpus(self) -> int:
        """Total number of CPU devices in mesh."""
        return len(self._cpus)

    def max_instances(self, gpus: int | None) -> int:
        """Calculate max instances given GPU requirement per instance.

        Args:
            gpus: GPUs required per instance (None = CPU-only)

        Returns:
            Maximum number of instances that can be deployed
        """
        if gpus is None or gpus == 0:
            return float("inf")  # No GPU limit
        return self.total_gpus // gpus

    def get_endpoints(self) -> list[str]:
        """Get list of endpoints from MeshNodes.

        Returns:
            List of endpoint addresses (e.g., ["server1:8000", "server2:8000"])
            Empty list if no endpoints configured
        """
        endpoints = []
        for node in self._nodes:
            if node.endpoint:
                endpoints.append(node.endpoint)
        return endpoints

    def __enter__(self):
        """Enter mesh context."""
        stack = _get_mesh_stack()
        stack.append(self)
        return self

    def __exit__(self, *args):
        """Exit mesh context."""
        stack = _get_mesh_stack()
        stack.pop()

    def __repr__(self):
        return f"Mesh(gpus={self.total_gpus}, cpus={self.total_cpus}, backend={self.backend})"


def get_current_mesh() -> Mesh | None:
    """Get currently active mesh from context.

    Returns:
        Current mesh if in a mesh context, None otherwise
    """
    stack = _get_mesh_stack()
    return stack[-1] if stack else None
