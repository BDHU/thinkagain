"""Shared utilities for backends."""

import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class RoundRobinPool:
    """Round-robin load balancer for instance pools.

    Manages multiple named pools of instances and distributes
    requests across instances using round-robin selection.
    """

    def __init__(self):
        self._instances: dict[str, list] = {}
        self._indices: dict[str, int] = {}
        self._lock = threading.Lock()

    def create_pool(self, name: str, instances: list) -> None:
        """Create a new pool of instances.

        Args:
            name: Pool identifier
            instances: List of instances to manage
        """
        with self._lock:
            self._instances[name] = instances
            self._indices[name] = 0

    def remove_pool(self, name: str) -> None:
        """Remove a pool.

        Args:
            name: Pool identifier to remove
        """
        with self._lock:
            self._instances.pop(name, None)
            self._indices.pop(name, None)

    def get_next(self, name: str) -> Any:
        """Get next instance using round-robin.

        Args:
            name: Pool identifier

        Returns:
            Next instance in the pool

        Raises:
            RuntimeError: If pool doesn't exist or is empty
        """
        with self._lock:
            if name not in self._instances or not self._instances[name]:
                raise RuntimeError(f"No instances in pool '{name}'")

            instances = self._instances[name]
            idx = self._indices[name]
            self._indices[name] = (idx + 1) % len(instances)
            return instances[idx]

    def has_pool(self, name: str) -> bool:
        """Check if pool exists.

        Args:
            name: Pool identifier

        Returns:
            True if pool exists, False otherwise
        """
        with self._lock:
            return name in self._instances

    def clear(self) -> None:
        """Clear all pools."""
        with self._lock:
            self._instances.clear()
            self._indices.clear()


def build_instances(cls: type, n: int, args: tuple, kwargs: dict) -> list[Any]:
    """Create initialized instances for a replica class."""
    initializer = getattr(cls, "__local_init__", cls)
    return [initializer(*args, **kwargs) for _ in range(n)]


class PoolBackendMixin:
    """Mixin for backends that use RoundRobinPool for instance management.

    Subclasses must initialize self._pool = RoundRobinPool() in __init__.
    """

    _pool: RoundRobinPool

    def _deploy_to_pool(
        self, name: str, cls: type, n: int, args: tuple, kwargs: dict
    ) -> None:
        """Deploy instances to the pool."""
        if self._pool.has_pool(name):
            return
        instances = build_instances(cls, n, args, kwargs)
        self._pool.create_pool(name, instances)

    def _shutdown_from_pool(self, name: str) -> None:
        """Remove instances from the pool."""
        self._pool.remove_pool(name)

    def _get_from_pool(self, name: str) -> Any:
        """Get next instance from pool via round-robin."""
        return self._pool.get_next(name)

    def _is_deployed_in_pool(self, name: str) -> bool:
        """Check if instances are deployed in the pool."""
        return self._pool.has_pool(name)
