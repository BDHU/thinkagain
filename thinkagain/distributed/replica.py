"""Replica specification for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import get_backend as _get_backend


@dataclass
class ReplicaSpec:
    """Specification for a replica class."""

    cls: type
    n: int = 1

    @property
    def name(self) -> str:
        return self.cls.__name__

    @property
    def full_name(self) -> str:
        return f"{self.cls.__module__}.{self.cls.__qualname__}"

    async def deploy(self, *args, **kwargs) -> None:
        """Deploy instances via the configured backend (async)."""
        await _get_backend().deploy(self, *args, **kwargs)

    async def shutdown(self) -> None:
        """Shutdown instances via the configured backend (async)."""
        await _get_backend().shutdown(self)

    def get(self) -> Any:
        """Get next instance via round-robin (sync).

        Raises:
            RuntimeError: If replica is not deployed.
        """
        backend = _get_backend()
        if not backend.is_deployed(self):
            raise RuntimeError(
                f"Replica '{self.name}' not deployed. "
                f"Call 'await {self.name}.deploy()' first."
            )

        instance = backend.get_instance(self)

        # Optional profiling hook (decoupled - only imports if profiling is enabled)
        try:
            from .profiling import record_replica_call

            record_replica_call(self.name)
        except (ImportError, AttributeError):
            pass

        return instance
