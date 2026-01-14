"""Mesh configuration for distributed execution.

This module provides mesh configuration for distributed execution across
multiple nodes. Mesh defines available resources and is used as an execution context.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .devices import CpuDevice, Device, GpuDevice

if TYPE_CHECKING:
    from ..runtime.runtime import RuntimeFactory


_mesh_stack: contextvars.ContextVar[tuple["Mesh", ...]] = contextvars.ContextVar(
    "mesh_stack", default=()
)


def _get_mesh_stack() -> tuple["Mesh", ...]:
    """Get context-local mesh stack."""
    return _mesh_stack.get()


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
        devices = [GpuDevice(id=i) for i in range(self.gpus)]
        if self.cpus > 0:
            devices.append(CpuDevice(id=0))
        return devices

    def __repr__(self):
        return f"MeshNode({self.name}, gpus={self.gpus}, cpus={self.cpus})"


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


class Mesh:
    """Resource mesh defining available compute for distributed execution.

    Mesh is used as a context manager to define the execution environment
    for distributed services. It specifies available devices (GPUs, CPUs)
    across single or multiple nodes.

    Example:
        # Local execution with auto-detection
        from thinkagain.resources.devices import devices
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
        runtime_factory: "RuntimeFactory | None" = None,
    ):
        """Initialize mesh with devices.

        Args:
            devices: List of Device or MeshNode objects
            backend: Backend type ("local" or "grpc")
        """
        self.backend = backend
        self._nodes = []  # Store original MeshNodes for endpoint tracking
        self._runtime_factory = runtime_factory

        expanded = self._expand_devices(devices)
        self.devices = expanded
        self._gpus = [d for d in self.devices if isinstance(d, GpuDevice)]
        self._cpus = [d for d in self.devices if isinstance(d, CpuDevice)]

    def _expand_devices(self, devices: list[Device | MeshNode]) -> list[Device]:
        expanded: list[Device] = []
        for device in devices:
            if isinstance(device, MeshNode):
                self._nodes.append(device)
                expanded.extend(device.expand())
            else:
                expanded.append(device)
        return expanded

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in mesh."""
        return len(self._gpus)

    @property
    def total_cpus(self) -> int:
        """Total number of CPU devices in mesh."""
        return len(self._cpus)

    def max_instances(self, gpus: int | None) -> int | float:
        """Calculate max instances given GPU requirement per instance.

        Args:
            gpus: GPUs required per instance (None = CPU-only)

        Returns:
            Maximum number of instances that can be deployed (float('inf') for CPU-only)
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
        return [node.endpoint for node in self._nodes if node.endpoint]

    async def _ensure_deployed(self, handle):
        """Ensure service is deployed (auto-deploy with n=1 if needed).

        Args:
            handle: ServiceHandle to deploy

        This is called automatically when a service is first used in a pipeline.
        """
        from ..runtime.pool import ensure_deployed, get_or_create_handle_pool

        pool = get_or_create_handle_pool(handle, self)
        await ensure_deployed(pool)

    def get_service_instance(self, handle):
        """Get next available instance for a service (round-robin).

        Args:
            handle: ServiceHandle

        Returns:
            Service instance
        """
        from ..runtime.pool import get_or_create_handle_pool

        pool = get_or_create_handle_pool(handle, self)
        if not pool._deployed:
            raise RuntimeError(
                f"Service {handle.service_class_name} not deployed. "
                f"This should not happen (auto-deploy failed)."
            )

        return pool.get_next()

    def __enter__(self):
        """Enter mesh context.

        Provides service execution capability for graph execution and sets up
        the DAGScheduler for dynamic task execution.
        """
        from ..backends.service_provider import ServiceExecutionProvider
        from ..runtime.context import ExecutionContext, set_current_execution_context
        from ..runtime.runtime import RuntimeFactory, set_current_runtime
        from ..runtime.scheduler import set_current_scheduler

        stack = _get_mesh_stack()
        self._mesh_token = _mesh_stack.set((*stack, self))
        self._execution_context_token = set_current_execution_context(
            ExecutionContext()
        )

        # Store service provider for execution
        self._service_provider = ServiceExecutionProvider(self)

        # Create and start runtime for dynamic execution
        import asyncio

        factory = self._runtime_factory or RuntimeFactory()
        self._runtime = factory.create_runtime(self)
        self._scheduler = self._runtime.scheduler
        set_current_scheduler(self._scheduler)
        self._runtime_token = set_current_runtime(self._runtime)

        # Start the scheduler's background loop
        # We need to do this in a non-blocking way
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._scheduler.start())
        except RuntimeError:
            # No event loop running, scheduler will start on first use
            pass

        return self

    def __exit__(self, *args):
        """Exit mesh context and stop the scheduler."""
        import asyncio

        from ..runtime.context import reset_current_execution_context
        from ..runtime.scheduler import set_current_scheduler
        from ..runtime.runtime import reset_current_runtime

        # Stop the scheduler
        scheduler = getattr(self, "_scheduler", None)
        if scheduler:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(scheduler.stop())
            except RuntimeError:
                pass
            set_current_scheduler(None)
            runtime_token = getattr(self, "_runtime_token", None)
            if runtime_token is not None:
                reset_current_runtime(runtime_token)
                self._runtime_token = None

        token = getattr(self, "_mesh_token", None)
        if token is not None:
            _mesh_stack.reset(token)
            self._mesh_token = None

        execution_token = getattr(self, "_execution_context_token", None)
        if execution_token is not None:
            reset_current_execution_context(execution_token)
            self._execution_context_token = None

    def get_service_provider(self):
        """Get the service provider for this mesh.

        Returns:
            ServiceExecutionProvider that can execute service calls
        """
        return getattr(self, "_service_provider", None)

    def __repr__(self):
        return f"Mesh(gpus={self.total_gpus}, cpus={self.total_cpus}, backend={self.backend})"


def get_current_mesh() -> Mesh | None:
    """Get currently active mesh from context.

    Returns:
        Current mesh if in a mesh context, None otherwise
    """
    stack = _get_mesh_stack()
    return stack[-1] if stack else None


def require_mesh(context: str = "This operation") -> Mesh:
    """Get current mesh or raise a consistent mesh-required error."""
    mesh = get_current_mesh()
    if mesh is None:
        raise RuntimeError(
            f"{context} requires a mesh context. "
            "Use 'with Mesh([...]):' to create a mesh before calling."
        )
    return mesh
