"""In-process backend for running replicas locally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec

from .utils import RoundRobinPool


class LocalBackend:
    """Local backend keeps replica instances within the same process."""

    def __init__(self):
        self._pool = RoundRobinPool()

    def deploy(self, spec: "ReplicaSpec", *args, **kwargs) -> None:
        name = spec.cls.__name__
        if self._pool.has_pool(name):
            return

        initializer = getattr(spec.cls, "__local_init__", spec.cls)
        instances = [initializer(*args, **kwargs) for _ in range(spec.n)]
        self._pool.create_pool(name, instances)

    def shutdown(self, spec: "ReplicaSpec") -> None:
        self._pool.remove_pool(spec.cls.__name__)

    def get_instance(self, spec: "ReplicaSpec") -> Any:
        return self._pool.get_next(spec.cls.__name__)

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        return self._pool.has_pool(spec.cls.__name__)
