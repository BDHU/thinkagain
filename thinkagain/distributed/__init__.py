"""Distributed execution components."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from .manager import (
    ReplicaHandle,
    ReplicaManager,
    get_default_manager,
    replica,
    set_default_manager,
)
from .replica import ReplicaSpec
from .runtime import (
    get_runtime_config,
    init,
    list_backends,
    register_backend,
    reset_backend,
)

if TYPE_CHECKING:
    from .backend.serialization import Serializer


@contextmanager
def runtime(
    backend: str = "local",
    address: str | None = None,
    *,
    manager: ReplicaManager | None = None,
    serializer: "Serializer" | None = None,
    **options,
):
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
    active_manager = manager or get_default_manager()
    init(backend=backend, address=address, serializer=serializer, **options)
    active_manager.deploy_all()
    try:
        yield
    finally:
        active_manager.shutdown_all()


__all__ = [
    "ReplicaManager",
    "ReplicaHandle",
    "ReplicaSpec",
    "replica",
    "get_default_manager",
    "set_default_manager",
    "init",
    "get_runtime_config",
    "list_backends",
    "register_backend",
    "reset_backend",
    "runtime",
]
