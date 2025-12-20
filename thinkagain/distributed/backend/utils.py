"""Shared utilities for backends."""

import threading
from typing import Any


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
