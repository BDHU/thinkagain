"""Op - representation of a unit of computation in the dynamic DAG."""

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from .object_ref import ObjectRef
from .utils import extract_object_refs

__all__ = ["Op", "ServiceOp"]


@dataclass
class Op:
    """Represents a single unit of computation to be executed.

    An Op captures a function call along with its arguments, which may include
    ObjectRefs representing dependencies on other ops. The scheduler uses this
    to build and execute the dynamic computation DAG.

    Attributes:
        op_id: Unique identifier for this op
        fn: The function or method to execute
        args: Positional arguments (may contain ObjectRefs)
        kwargs: Keyword arguments (may contain ObjectRefs)
        result_ref: ObjectRef where the result will be stored
    """

    op_id: UUID
    fn: Callable | None = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    result_ref: ObjectRef | None = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.kwargs is None:
            self.kwargs = {}

    def get_dependencies(self) -> list[ObjectRef]:
        """Extract all ObjectRef dependencies from args and kwargs.

        Returns:
            List of ObjectRefs that this task depends on
        """
        deps: list[ObjectRef] = []
        for arg in self.args:
            deps.extend(extract_object_refs(arg))
        for value in self.kwargs.values():
            deps.extend(extract_object_refs(value))
        return deps

    def __repr__(self) -> str:
        """Human-readable representation."""
        fn_name = getattr(self.fn, "__name__", str(self.fn))
        return f"Op({self.op_id}, fn={fn_name}, deps={len(self.get_dependencies())})"


@dataclass
class ServiceOp(Op):
    """Op representing a method call on a service instance.

    Extends Op with service-specific information for routing the call
    to the correct service instance and maintaining service state.

    Attributes:
        service_handle: Handle to the service instance
        method_name: Name of the method to call on the service
    """

    service_handle: Any = None  # Service (avoid circular import)
    method_name: str = ""

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ServiceOp({self.op_id}, "
            f"service={self.service_handle.service_class.__name__}, "
            f"method={self.method_name})"
        )
