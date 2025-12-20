"""Runtime configuration and backend selection for replicas."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from .backend.base import Backend
from .backend.local import LocalBackend
from .backend.serialization import Serializer


class RuntimeContext:
    """Thread-safe runtime context for backend management.

    Manages runtime configuration and thread-local backend instances
    to support concurrent usage safely.
    """

    def __init__(self):
        self._local = threading.local()
        self._config: dict[str, Any] = {
            "backend": "local",
            "address": None,
            "options": {},
            "serializer": None,
        }
        self._lock = threading.RLock()

    @property
    def config(self) -> dict[str, Any]:
        """Get current configuration (copy for safety)."""
        with self._lock:
            return dict(self._config)

    def init(
        self,
        backend: str = "local",
        address: str | None = None,
        *,
        serializer: Serializer | None = None,
        **options,
    ) -> None:
        """Initialize runtime configuration (thread-safe)."""
        with self._lock:
            self._config = {
                "backend": backend,
                "address": address,
                "options": options,
                "serializer": serializer,
            }
            # Clear thread-local backend to force recreation
            if hasattr(self._local, "backend"):
                self._local.backend = None

    def get_backend(self) -> Backend:
        """Get thread-local backend instance (thread-safe)."""
        # Fast path - no lock for existing backend
        if hasattr(self._local, "backend") and self._local.backend is not None:
            return self._local.backend

        # Slow path - create backend with lock
        with self._lock:
            # Double-check - another thread may have created it
            if not hasattr(self._local, "backend") or self._local.backend is None:
                self._local.backend = self._create_backend()
            return self._local.backend

    def _create_backend(self) -> Backend:
        """Create backend instance from config (called within lock)."""
        backend_type = self._config["backend"]

        def _grpc_backend() -> Backend:
            from .backend.grpc import GrpcBackend

            return GrpcBackend(
                self._config["address"],
                self._config["options"],
                serializer=self._config["serializer"],
            )

        factories = {
            "local": LocalBackend,
            "grpc": _grpc_backend,
        }

        try:
            factory = factories[backend_type]
        except KeyError as exc:
            raise ValueError(f"Unknown backend: {backend_type}") from exc

        return factory()

    def reset_backend(self) -> None:
        """Reset thread-local backend (thread-safe)."""
        if hasattr(self._local, "backend"):
            self._local.backend = None

    def restore_config(self, config: dict[str, Any]) -> None:
        """Restore configuration from a previous state (thread-safe).

        Args:
            config: Configuration dict to restore
        """
        with self._lock:
            self._config = dict(config)  # Make a copy for safety
            # Clear thread-local backend to force recreation
            if hasattr(self._local, "backend"):
                self._local.backend = None


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


def get_runtime_config() -> dict[str, Any]:
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

    Args:
        backend: Backend type ("local" or "grpc")
        address: Address for gRPC backend
        serializer: Custom serializer for gRPC
        **options: Additional backend options
    """
    old_config = _runtime.config
    try:
        _runtime.init(backend, address=address, serializer=serializer, **options)
        yield
    finally:
        _runtime.restore_config(old_config)
