"""Replica specification for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .profiling import record_replica_call
from .runtime import get_backend as _get_backend


@dataclass
class ReplicaSpec:
    """Specification for a replica class."""

    cls: type
    n: int = 1
    _deploy_args: tuple = field(default_factory=tuple)
    _deploy_kwargs: dict = field(default_factory=dict)

    def deploy_instances(self, *args, **kwargs) -> None:
        """Deploy instances via the configured backend."""
        if args or kwargs:
            self._deploy_args = args
            self._deploy_kwargs = kwargs
        _get_backend().deploy(self, *args, **kwargs)

    def shutdown_instances(self) -> None:
        """Shutdown instances via the configured backend."""
        _get_backend().shutdown(self)

    def get_instance(self) -> Any:
        """Get next instance, auto-deploying if needed."""
        backend = _get_backend()
        if not backend.is_deployed(self):
            self.deploy_instances(*self._deploy_args, **self._deploy_kwargs)

        # Record replica call for profiling (if enabled)
        record_replica_call(self.cls.__name__)

        return backend.get_instance(self)
