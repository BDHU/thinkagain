"""Runtime configuration and backend selection for replicas."""

from __future__ import annotations

from typing import Any

from .backend.base import Backend
from .backend.local import LocalBackend

# Runtime configuration shared across replicas
_runtime_config: dict[str, Any] = {
    "backend": "local",
    "address": None,
    "options": {},
}

# Singleton backend instance (created lazily)
_backend: Backend | None = None


def init(backend: str = "local", address: str | None = None, **options) -> None:
    """Initialize the distributed runtime."""
    _runtime_config["backend"] = backend
    _runtime_config["address"] = address
    _runtime_config["options"] = options


def get_runtime_config() -> dict[str, Any]:
    """Get the current runtime configuration."""
    return dict(_runtime_config)


def get_backend() -> Backend:
    """Instantiate or return the configured backend."""
    global _backend
    if _backend is not None:
        return _backend

    backend_type = _runtime_config["backend"]
    if backend_type == "local":
        _backend = LocalBackend()
    elif backend_type == "grpc":
        from .backend.grpc import GrpcBackend

        _backend = GrpcBackend(
            _runtime_config["address"], _runtime_config["options"]
        )
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
    return _backend


def reset_backend() -> None:
    """Reset backend instance. Call this when changing backend configuration."""
    global _backend
    _backend = None
