"""Replica decorator and registry for distributed execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

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
        return backend.get_instance(self)


# Global registry of replica classes (thread-safe)
_replica_registry: dict[str, ReplicaSpec] = {}
_registry_lock = threading.RLock()


def _list_specs() -> list[ReplicaSpec]:
    """Snapshot replica specs under lock."""
    with _registry_lock:
        return list(_replica_registry.values())


def replica(_cls: type | None = None, *, n: int = 1):
    """Mark a class as a distributable replica pool.

    Replicas are stateful service pools. Access instances via Replica.get().

    Args:
        n: Number of instances to create when deployed.

    Supports both @replica and @replica(n=2) styles.

    Example:
        @replica(n=4)
        class LLMPool:
            def __init__(self, model_name):
                self.model = load_model(model_name)

            def invoke(self, prompt):
                return self.model.generate(prompt)

        # In a node:
        @node
        async def generate(ctx):
            llm = LLMPool.get()  # Returns instance via round-robin
            response = llm.invoke(ctx.get("prompt"))
            ctx.set("response", response)
            return ctx
    """

    def decorator(cls: type) -> type:
        spec = ReplicaSpec(cls=cls, n=n)
        with _registry_lock:
            _replica_registry[cls.__name__] = spec

        @classmethod
        def deploy(cls, *args, **kwargs):
            """Deploy instances. Idempotent; call shutdown() to redeploy."""
            spec.deploy_instances(*args, **kwargs)

        @classmethod
        def shutdown(cls):
            """Shutdown all instances."""
            spec.shutdown_instances()

        @classmethod
        def get(cls) -> Any:
            """Get next instance via round-robin."""
            return spec.get_instance()

        cls.deploy = deploy
        cls.shutdown = shutdown
        cls.get = get

        return cls

    if _cls is None:
        return decorator

    return decorator(_cls)


def get_replica_spec(name: str) -> ReplicaSpec | None:
    """Get replica spec by class name (thread-safe)."""
    with _registry_lock:
        return _replica_registry.get(name)


def get_all_replicas() -> dict[str, ReplicaSpec]:
    """Get all registered replicas (thread-safe)."""
    with _registry_lock:
        return dict(_replica_registry)


def clear_replica_registry() -> None:
    """Clear the replica registry (thread-safe). Useful for testing."""
    with _registry_lock:
        _replica_registry.clear()


def deploy() -> None:
    """Deploy all registered replicas with default constructor (thread-safe).

    Idempotent: only deploys replicas that haven't been deployed yet.
    Use ReplicaClass.deploy(...) for custom constructor arguments.
    """
    for spec in _list_specs():
        spec.deploy_instances()


def shutdown() -> None:
    """Shutdown all replica instances (thread-safe)."""
    for spec in _list_specs():
        spec.shutdown_instances()
