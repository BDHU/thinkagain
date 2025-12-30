"""Runtime configuration and backend selection for replicas."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .backend.base import Backend
from .backend.local import LocalBackend
from .backend.serialization import Serializer

_BACKEND_FACTORIES: dict[str, Callable[["RuntimeConfig"], Backend]] = {}


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
    """Thread-safe runtime context for backend management."""

    def __init__(self):
        self._local = threading.local()
        self._config = RuntimeConfig()
        self._config_version = 0
        self._lock = threading.Lock()

    @property
    def config(self) -> RuntimeConfig:
        """Get current configuration (copy for safety)."""
        with self._lock:
            return RuntimeConfig(
                backend=self._config.backend,
                address=self._config.address,
                options=dict(self._config.options),
                serializer=self._config.serializer,
            )

    def set_config(self, config: RuntimeConfig) -> None:
        """Set configuration and clear cached backend."""
        with self._lock:
            self._config = config
            self._config_version += 1
            self._local.backend = None
            self._local.backend_version = None

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
        """Get thread-local backend instance."""
        backend = getattr(self._local, "backend", None)
        backend_version = getattr(self._local, "backend_version", None)

        # Fast path: return cached backend if version matches
        if backend is not None and backend_version == self._config_version:
            return backend

        # Slow path: create or update backend under lock
        with self._lock:
            # Double-check version after acquiring lock
            if backend is None or backend_version != self._config_version:
                backend = self._create_backend()
                self._local.backend = backend
                self._local.backend_version = self._config_version
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
        """Reset thread-local backend."""
        self._local.backend = None
        self._local.backend_version = None

    def close_backend(self) -> None:
        """Close the current thread-local backend if it supports close()."""
        backend = getattr(self._local, "backend", None)
        if backend is not None and hasattr(backend, "close"):
            backend.close()


# Global runtime instance (thread-safe)
_runtime = RuntimeContext()


def init(
    backend: str = "local",
    address: str | None = None,
    *,
    serializer: Serializer | None = None,
    **options,
) -> None:
    """Initialize the distributed runtime (thread-safe).

    Args:
        backend: Backend type ("local" or "grpc")
        address: Address for gRPC backend
        serializer: Custom serializer for gRPC
        **options: Additional backend options
    """
    _runtime.init(backend, address=address, serializer=serializer, **options)


def get_runtime_config() -> RuntimeConfig:
    """Get the current runtime configuration (thread-safe)."""
    return _runtime.config


def get_backend() -> Backend:
    """Get the current backend instance (thread-safe).

    Returns thread-local backend, creating it if needed.
    """
    return _runtime.get_backend()


def reset_backend() -> None:
    """Reset the backend instance (thread-safe).

    Forces recreation on next get_backend() call.
    Useful for testing or reconfiguration.
    """
    _runtime.reset_backend()


@contextmanager
def runtime(
    backend: str = "local",
    address: str | None = None,
    *,
    serializer: Serializer | None = None,
    **options,
) -> Iterator[None]:
    """Context manager for scoped runtime configuration.

    Useful for testing - automatically restores previous config on exit.

    Example:
        with runtime(backend="local"):
            # ... test code with local backend ...
            pass
        # Previous config restored
    """
    old_config = _runtime.config
    try:
        _runtime.init(backend, address=address, serializer=serializer, **options)
        yield
    finally:
        _runtime.close_backend()
        _runtime.set_config(old_config)


def _default_local_factory(config: RuntimeConfig) -> Backend:
    return LocalBackend()


def _default_grpc_factory(config: RuntimeConfig) -> Backend:
    from .backend.grpc import GrpcBackend

    return GrpcBackend(
        config.address,
        dict(config.options),
        serializer=config.serializer,
    )


register_backend("local", _default_local_factory)
register_backend("grpc", _default_grpc_factory)
