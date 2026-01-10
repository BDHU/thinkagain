"""Replication module for distributed execution."""

from .pool import (
    LocalReplica,
    RemoteReplica,
    Replica,
    ReplicaPool,
    get_or_create_pool,
    shutdown_all,
)

__all__ = [
    "Replica",
    "LocalReplica",
    "RemoteReplica",
    "ReplicaPool",
    "get_or_create_pool",
    "shutdown_all",
]
