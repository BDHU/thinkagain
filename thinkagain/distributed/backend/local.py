"""In-process backend for running replicas locally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thinkagain.distributed.replica import ReplicaSpec


class LocalBackend:
    """Local backend keeps replica instances within the same process."""

    def __init__(self):
        self._instances: dict[str, list] = {}
        self._round_robin_idx: dict[str, int] = {}

    def deploy(self, spec: "ReplicaSpec", *args, **kwargs) -> None:
        name = spec.cls.__name__
        if name in self._instances:
            return
        self._instances[name] = [spec.cls(*args, **kwargs) for _ in range(spec.n)]
        self._round_robin_idx[name] = 0

    def shutdown(self, spec: "ReplicaSpec") -> None:
        name = spec.cls.__name__
        if name in self._instances:
            del self._instances[name]
            del self._round_robin_idx[name]

    def get_instance(self, spec: "ReplicaSpec") -> Any:
        name = spec.cls.__name__
        instances = self._instances[name]
        idx = self._round_robin_idx[name]
        self._round_robin_idx[name] = (idx + 1) % len(instances)
        return instances[idx]

    def is_deployed(self, spec: "ReplicaSpec") -> bool:
        return spec.cls.__name__ in self._instances
