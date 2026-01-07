"""Node configuration and state management for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeConfig:
    """Configuration for a compute node.

    Examples:
        Local node:
            NodeConfig(host="localhost", cpus=16, gpus=2)

        Remote node:
            NodeConfig(
                host="worker1.example.com",
                cpus=32,
                gpus=4,
                ssh_user="ubuntu",
                ssh_key_path="~/.ssh/id_rsa",
                env_setup="source /opt/venv/bin/activate",
            )
    """

    host: str
    cpus: int
    gpus: int

    # SSH configuration for remote spawning
    ssh_port: int = 22
    ssh_user: str | None = None
    ssh_key_path: str | None = None

    # Remote environment configuration
    python_executable: str = "python3"  # Python command on remote node
    env_setup: str | None = (
        None  # Command to activate environment (e.g., "source venv/bin/activate")
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.cpus < 0 or self.gpus < 0:
            raise ValueError("Resource counts cannot be negative")
        if self.cpus == 0 and self.gpus == 0:
            raise ValueError(
                "At least one resource must be > 0. "
                "Specify cpus=N for CPU-only or gpus=N for GPU-only."
            )

    @property
    def is_local(self) -> bool:
        """Check if this is a localhost node."""
        return self.host in ("localhost", "127.0.0.1", "::1")


@dataclass
class NodeState:
    """Runtime state tracking for a node."""

    config: NodeConfig
    available_cpus: int
    available_gpus: int
    server_count: int = 0

    @classmethod
    def from_config(cls, config: NodeConfig) -> "NodeState":
        """Create initial state from config."""
        return cls(
            config=config,
            available_cpus=config.cpus,
            available_gpus=config.gpus,
        )

    def can_fit(self, cpus: int, gpus: int) -> bool:
        """Check if resources are available."""
        return self.available_cpus >= cpus and self.available_gpus >= gpus

    def reserve(self, cpus: int, gpus: int) -> None:
        """Reserve resources (decrease available)."""
        if not self.can_fit(cpus, gpus):
            raise RuntimeError(
                f"Cannot reserve {cpus} CPUs, {gpus} GPUs on {self.config.host}: "
                f"only {self.available_cpus} CPUs, {self.available_gpus} GPUs available"
            )
        self.available_cpus -= cpus
        self.available_gpus -= gpus
        self.server_count += 1

    def release(self, cpus: int, gpus: int) -> None:
        """Release resources (increase available)."""
        self.available_cpus += cpus
        self.available_gpus += gpus
        self.server_count -= 1

    @property
    def utilization(self) -> dict[str, float]:
        """Get resource utilization percentages."""
        total_cpus = self.config.cpus
        total_gpus = self.config.gpus
        return {
            "cpu": (1.0 - self.available_cpus / total_cpus) if total_cpus > 0 else 0.0,
            "gpu": (1.0 - self.available_gpus / total_gpus) if total_gpus > 0 else 0.0,
        }
