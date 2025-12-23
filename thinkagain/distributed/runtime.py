"""Runtime configuration and backend selection for replicas."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from .backend.base import Backend
from .backend.local import LocalBackend
from .backend.serialization import Serializer


class RuntimeContext:
    """Thread-safe runtime context for backend management."""

    def __init__(self):
        self._local = threading.local()
        self._config: dict[str, Any] = {
            "backend": "local",
            "address": None,
            "options": {},
            "serializer": None,
        }
        self._lock = threading.Lock()

    @property
    def config(self) -> dict[str, Any]:
        """Get current configuration (copy for safety)."""
        with self._lock:
            return dict(self._config)

    def _set_config(self, config: dict[str, Any]) -> None:
        """Set configuration and clear cached backend."""
        with self._lock:
            self._config = config
            self._local.backend = None

    def init(
        self,
        backend: str = "local",
        address: str | None = None,
        *,
        serializer: Serializer | None = None,
        **options,
    ) -> None:
        """Initialize runtime configuration."""
        self._set_config(
            {
                "backend": backend,
                "address": address,
                "options": options,
                "serializer": serializer,
            }
        )

    def get_backend(self) -> Backend:
        """Get thread-local backend instance."""
        backend = getattr(self._local, "backend", None)
        if backend is None:
            with self._lock:
                backend = self._create_backend()
                self._local.backend = backend
        return backend

    def _create_backend(self) -> Backend:
        """Create backend instance from config."""
        backend_type = self._config["backend"]
        if backend_type == "local":
            return LocalBackend()
        elif backend_type == "grpc":
            from .backend.grpc import GrpcBackend

            return GrpcBackend(
                self._config["address"],
                self._config["options"],
                serializer=self._config["serializer"],
            )
        raise ValueError(f"Unknown backend: {backend_type}")

    def reset_backend(self) -> None:
        """Reset thread-local backend."""
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
    """
    old_config = _runtime.config
    try:
        _runtime.init(backend, address=address, serializer=serializer, **options)
        yield
    finally:
        current_backend = getattr(_runtime._local, "backend", None)
        if current_backend is not None and hasattr(current_backend, "close"):
            current_backend.close()
        _runtime._set_config(old_config)
