"""Runtime configuration and backend selection for replicas."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TYPE_CHECKING

from .backend.base import Backend
from .backend.local import LocalBackend
from .backend.serialization import Serializer
from .nodes import NodeConfig

_BACKEND_FACTORIES: dict[str, Callable[["RuntimeConfig"], Backend]] = {}

if TYPE_CHECKING:
    from .manager import ReplicaManager


def register_backend(name: str, factory: Callable[["RuntimeConfig"], Backend]) -> None:
    """Register a backend factory by name."""
    _BACKEND_FACTORIES[name] = factory


def get_backend_factory(
    name: str,
) -> Callable[["RuntimeConfig"], Backend] | None:
    """Lookup a backend factory by name."""
    return _BACKEND_FACTORIES.get(name)


def list_backends() -> list[str]:
    """List registered backend names."""
    return sorted(_BACKEND_FACTORIES)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for backend selection."""

    backend: str = "local"
    address: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    serializer: Serializer | None = None


class RuntimeContext:
    """Runtime context for backend management."""

    def __init__(self):
        self._config = RuntimeConfig()
        self._config_version = 0
        self._backend: Backend | None = None
        self._backend_version: int | None = None

    @property
    def config(self) -> RuntimeConfig:
        """Get current configuration (copy for safety)."""
        return replace(self._config, options=dict(self._config.options))

    def set_config(self, config: RuntimeConfig) -> None:
        """Set configuration and clear cached backend."""
        self._config = config
        self._config_version += 1
        self._backend = None
        self._backend_version = None

    def init(
        self,
        backend: str = "local",
        address: str | None = None,
        *,
        serializer: Serializer | None = None,
        **options,
    ) -> None:
        """Initialize runtime configuration."""
        self.set_config(
            RuntimeConfig(
                backend=backend,
                address=address,
                options=dict(options),
                serializer=serializer,
            )
        )

    def get_backend(self) -> Backend:
        """Get cached backend instance."""
        backend = self._backend
        backend_version = self._backend_version

        # Fast path: return cached backend if version matches
        if backend is not None and backend_version == self._config_version:
            return backend

        # Slow path: create or update backend
        if backend is None or backend_version != self._config_version:
            backend = self._create_backend()
            self._backend = backend
            self._backend_version = self._config_version
        return backend

    def _create_backend(self) -> Backend:
        """Create backend instance from config."""
        backend_type = self._config.backend

        factory = get_backend_factory(backend_type)
        if factory is None:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Registered: {', '.join(list_backends()) or 'none'}"
            )
        return factory(self._config)

    def reset_backend(self) -> None:
        """Reset cached backend."""
        self._backend = None
        self._backend_version = None

    def close_backend(self) -> None:
        """Close the current backend if it supports close()."""
        backend = self._backend
        if backend is not None and hasattr(backend, "close"):
            backend.close()


# Global runtime instance (thread-safe)
_runtime = RuntimeContext()


def init(
    backend: str = "local",
    *,
    nodes: list[NodeConfig] | None = None,
    serializer: Serializer | None = None,
    **options,
) -> None:
    """Initialize the distributed runtime.

    Args:
        backend: Backend type ("local" or "grpc")
        nodes: List of compute nodes (default: localhost with auto-detected CPUs)
        serializer: Custom serializer for gRPC
        **options: Additional backend options

    Examples:
        # Local with defaults
        init(backend="grpc")

        # Multi-node cluster
        init(
            backend="grpc",
            nodes=[
                NodeConfig(host="localhost", cpus=8, gpus=0),
                NodeConfig(host="worker1.local", cpus=16, gpus=2),
                NodeConfig(host="worker2.local", cpus=16, gpus=2),
            ],
        )
    """
    if nodes is not None:
        options["nodes"] = nodes
    _runtime.init(backend, address=None, serializer=serializer, **options)


def get_runtime_config() -> RuntimeConfig:
    """Get the current runtime configuration."""
    return _runtime.config


def get_backend() -> Backend:
    """Get the current backend instance.

    Returns thread-local backend, creating it if needed.
    """
    return _runtime.get_backend()


def reset_backend() -> None:
    """Reset the backend instance.

    Forces recreation on next get_backend() call.
    Useful for testing or reconfiguration.
    """
    _runtime.reset_backend()


@contextmanager
def runtime(
    backend: str = "local",
    address: str | None = None,
    *,
    manager: "ReplicaManager | None" = None,
    serializer: Serializer | None = None,
    **options,
) -> Iterator[None]:
    """Context manager for distributed replica lifecycle.

    Usage:
        from thinkagain import distributed

        with distributed.runtime():
            result = run(pipeline, data)

        # Or with explicit backend:
        with distributed.runtime(
            backend="grpc",
            address="localhost:50051",
        ):
            result = run(pipeline, data)
    """
    import asyncio
    from .manager import get_default_manager

    old_config = _runtime.config
    active_manager = manager or get_default_manager()
    try:
        _runtime.init(backend, address=address, serializer=serializer, **options)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(active_manager.deploy_all())
        except RuntimeError:
            asyncio.run(active_manager.deploy_all())
        yield
    finally:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(active_manager.shutdown_all())
        except RuntimeError:
            asyncio.run(active_manager.shutdown_all())
        _runtime.close_backend()
        _runtime.set_config(old_config)


register_backend("local", lambda _: LocalBackend())


def _grpc_factory(config: RuntimeConfig) -> Backend:
    from .backend.multinode_grpc import MultiNodeGrpcBackend

    return MultiNodeGrpcBackend(dict(config.options), serializer=config.serializer)


register_backend("grpc", _grpc_factory)
