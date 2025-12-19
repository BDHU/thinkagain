"""Distributed execution components."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from .replica import (
    ReplicaSpec,
    clear_replica_registry,
    deploy,
    get_all_replicas,
    get_replica_spec,
    replica,
    shutdown,
)
from .runtime import reset_backend, get_runtime_config, init

if TYPE_CHECKING:
    from .backend.serialization import Serializer


@contextmanager
def runtime(
    backend: str = "local",
    address: str | None = None,
    *,
    serializer: "Serializer" | None = None,
    **options,
):
    """Context manager for distributed replica lifecycle.

    Usage:
        from thinkagain import distributed

        with distributed.runtime():
            result = run(pipeline, data)

        # Or with explicit backend:
        with distributed.runtime(backend="grpc", address="localhost:50051"):
            result = run(pipeline, data)
    """
    init(backend=backend, address=address, serializer=serializer, **options)
    deploy()
    try:
        yield
    finally:
        shutdown()


__all__ = [
    "ReplicaSpec",
    "replica",
    "get_replica_spec",
    "get_all_replicas",
    "clear_replica_registry",
    "deploy",
    "shutdown",
    "init",
    "get_runtime_config",
    "reset_backend",
    "runtime",
]
