"""Task - representation of a unit of computation in the dynamic DAG."""

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from .object_ref import ObjectRef

__all__ = ["Task", "ActorTask"]


@dataclass
class Task:
    """Represents a single unit of computation to be executed.

    A Task captures a function call along with its arguments, which may include
    ObjectRefs representing dependencies on other tasks. The scheduler uses this
    to build and execute the dynamic computation DAG.

    Attributes:
        task_id: Unique identifier for this task
        fn: The function or method to execute
        args: Positional arguments (may contain ObjectRefs)
        kwargs: Keyword arguments (may contain ObjectRefs)
        result_ref: ObjectRef where the result will be stored
    """

    task_id: UUID
    fn: Callable
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result_ref: ObjectRef | None = None

    def get_dependencies(self) -> list[ObjectRef]:
        """Extract all ObjectRef dependencies from args and kwargs.

        Returns:
            List of ObjectRefs that this task depends on
        """
        deps: list[ObjectRef] = []

        # Check positional args
        for arg in self.args:
            if isinstance(arg, ObjectRef):
                deps.append(arg)
            elif isinstance(arg, (list, tuple)):
                deps.extend(self._extract_refs_from_container(arg))

        # Check keyword args
        for value in self.kwargs.values():
            if isinstance(value, ObjectRef):
                deps.append(value)
            elif isinstance(value, (list, tuple, dict)):
                deps.extend(self._extract_refs_from_container(value))

        return deps

    def _extract_refs_from_container(self, container: Any) -> list[ObjectRef]:
        """Recursively extract ObjectRefs from nested containers."""
        refs: list[ObjectRef] = []

        if isinstance(container, ObjectRef):
            refs.append(container)
        elif isinstance(container, (list, tuple)):
            for item in container:
                refs.extend(self._extract_refs_from_container(item))
        elif isinstance(container, dict):
            for value in container.values():
                refs.extend(self._extract_refs_from_container(value))

        return refs

    def __repr__(self) -> str:
        """Human-readable representation."""
        fn_name = getattr(self.fn, "__name__", str(self.fn))
        return (
            f"Task({self.task_id}, fn={fn_name}, deps={len(self.get_dependencies())})"
        )


@dataclass
class ActorTask(Task):
    """Task representing a method call on an actor replica.

    Extends Task with actor-specific information for routing the call
    to the correct actor instance and maintaining actor state.

    Attributes:
        actor_handle: Handle to the actor instance
        method_name: Name of the method to call on the actor
    """

    actor_handle: Any = None  # ActorHandle (avoid circular import)
    method_name: str = ""

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ActorTask({self.task_id}, "
            f"actor={self.actor_handle._replica_class.__name__}, "
            f"method={self.method_name})"
        )
