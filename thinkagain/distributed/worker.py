"""Worker decorator and registry for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class WorkerServiceError(AttributeError):
    """Raised when accessing a non-node method on a worker class."""

    pass


class WorkerMeta(type):
    """Metaclass that blocks access to non-node methods on worker classes."""

    def __getattribute__(cls, name: str):
        attr = super().__getattribute__(name)

        # Allow special attributes, private methods, and class methods
        if name.startswith("_") or name in ("launch", "shutdown"):
            return attr

        # Check if it's a callable (method) but not a Node
        if callable(attr) and not _is_node(attr):
            raise WorkerServiceError(
                f"'{name}' is not a @node service on worker '{cls.__name__}'. "
                f"Only @node methods can be called on worker classes."
            )

        return attr


def _is_node(obj) -> bool:
    """Check if an object is a Node."""
    # Import here to avoid circular imports
    from ..core.node import Node

    return isinstance(obj, Node)


@dataclass
class WorkerSpec:
    """Specification for a worker class."""

    cls: type
    n: int = 1
    _instances: list = field(default_factory=list)
    _round_robin_idx: int = 0

    def get_instance(self):
        """Get next instance using round-robin."""
        if not self._instances:
            raise RuntimeError(
                f"No instances available for worker '{self.cls.__name__}'. "
                f"Call {self.cls.__name__}.launch() first."
            )
        idx = self._round_robin_idx
        self._round_robin_idx = (idx + 1) % len(self._instances)
        return self._instances[idx]


# Global registry of worker classes
_worker_registry: dict[str, WorkerSpec] = {}


def worker(_cls: type | None = None, *, n: int = 1):
    """Mark a class as a distributable worker. Auto-registers on definition.

    Args:
        n: Number of instances to create when launched.

    Supports both @worker and @worker(n=2) styles.
    """

    def decorator(cls: type) -> type:
        # Create a new class with WorkerMeta as its metaclass
        # This allows us to intercept attribute access on the class
        new_cls = WorkerMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))

        spec = WorkerSpec(cls=new_cls, n=n)
        _worker_registry[new_cls.__name__] = spec

        # Add helper methods to the class
        @classmethod
        def launch(cls, instances: list | None = None):
            """Launch worker instances.

            Args:
                instances: Optional list of pre-created instances.
                          If not provided, creates n instances using default constructor.
            """
            spec = _worker_registry[cls.__name__]
            if instances is not None:
                spec._instances = list(instances)
            else:
                spec._instances = [cls() for _ in range(spec.n)]
            spec._round_robin_idx = 0

        @classmethod
        def shutdown(cls):
            """Clear worker instances."""
            spec = _worker_registry[cls.__name__]
            spec._instances = []
            spec._round_robin_idx = 0

        new_cls.launch = launch
        new_cls.shutdown = shutdown
        # Convenience attribute for introspection on instances
        new_cls.instances = property(
            lambda self: _worker_registry[new_cls.__class__.__name__]._instances
        )

        return new_cls

    if _cls is None:
        # Used as @worker(n=2)
        return decorator

    # Used as simple @worker
    return decorator(_cls)


def get_worker_spec(name: str) -> WorkerSpec | None:
    """Get worker spec by class name."""
    return _worker_registry.get(name)


def get_all_workers() -> dict[str, WorkerSpec]:
    """Get all registered workers."""
    return dict(_worker_registry)


def clear_worker_registry() -> None:
    """Clear the worker registry. Useful for testing."""
    _worker_registry.clear()


def init() -> None:
    """Initialize the distributed runtime by launching all registered workers.

    For each registered @worker class, creates n instances using the
    default constructor. Use WorkerClass.launch([...]) instead if you
    need custom initialization parameters.
    """
    for spec in _worker_registry.values():
        if not spec._instances:
            spec._instances = [spec.cls() for _ in range(spec.n)]
            spec._round_robin_idx = 0


def shutdown() -> None:
    """Shutdown all worker instances."""
    for spec in _worker_registry.values():
        spec._instances = []
        spec._round_robin_idx = 0
