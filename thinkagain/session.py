"""Session for distributed service execution with automatic optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .resources import Mesh


class Session:
    """Execution session with automatic optimization.

    The session manages service execution, resource allocation, and intelligent
    auto-scaling. When used as a context manager, it:

    1. Registers itself as the active execution context
    2. Starts background optimizer (future implementation)
    3. Monitors service metrics (queue depth, latency, throughput)
    4. Automatically scales service replicas to meet performance goals
    5. Handles resource contention when GPU capacity is exhausted

    The optimizer runs completely in the background - users just call
    service methods with .go() and scaling happens automatically.

    Attributes:
        mesh: Mesh configuration (compute resources)
        optimize: Optimization profile
            - "latency": Minimize latency (aggressive scaling, more replicas)
            - "throughput": Maximize throughput (optimize for high request rates)
            - "balanced": Balance latency and throughput (default)

    Examples:
        # Basic usage
        mesh = ta.Mesh(devices=[ta.GpuDevice(i) for i in range(8)])
        session = ta.Session(mesh=mesh, optimize="balanced")

        llm = LLM.init()
        with session:
            result = await llm.generate.go("Hello")

        # Multiple sessions for different environments
        dev_session = ta.Session(
            mesh=ta.Mesh(devices=[ta.GpuDevice(0)]),
            optimize="throughput",
        )

        prod_session = ta.Session(
            mesh=ta.Mesh(devices=[ta.GpuDevice(i) for i in range(16)]),
            optimize="latency",
        )
    """

    def __init__(
        self,
        mesh: Mesh,
        optimize: Literal["latency", "throughput", "balanced"] = "balanced",
    ):
        """Create session.

        Args:
            mesh: Mesh configuration (devices)
            optimize: Optimization profile
                - "latency": Minimize latency (aggressive scaling)
                - "throughput": Maximize throughput (optimize for high request rates)
                - "balanced": Balance latency and throughput
        """
        self.mesh = mesh
        self.optimize = optimize
        self._mesh_token = None

    def __enter__(self):
        """Start session.

        This delegates to the mesh's __enter__ method, which sets up
        the runtime and scheduler. The session is kept isolated from
        the core graph execution logic.
        """
        self.mesh.__enter__()
        return self

    def __exit__(self, *args):
        """Stop session."""
        self.mesh.__exit__(*args)
