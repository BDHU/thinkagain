"""RemoteFunction - wrapper that adds .go() method to @node functions."""

from typing import Any, Callable, TypeVar
from uuid import uuid4

from ..runtime.utils import maybe_await
from ..runtime.object_ref import ObjectRef
from ..runtime import get_current_runtime
from ..runtime.task import Task

__all__ = ["RemoteFunction", "wrap_as_remote_function"]

T = TypeVar("T")


class RemoteFunction:
    """Wrapper that adds .go() method to functions.

    This allows functions to be called in two ways:
    1. Direct call: await fn(x) - executes immediately (bypasses scheduler)
    2. Remote call: fn.go(x) - submits to scheduler, returns ObjectRef

    Example:
        @node
        async def process(x: str) -> str:
            return x.upper()

        # Direct execution
        result = await process("hello")

        # Remote execution (via scheduler)
        ref = process.go("hello")
        result = await ref
    """

    def __init__(self, fn: Callable):
        """Initialize the remote function wrapper.

        Args:
            fn: The function to wrap
        """
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "remote_function")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", None)
        self.__qualname__ = getattr(fn, "__qualname__", None)

        # Copy over special attributes from original function
        if hasattr(fn, "_is_node"):
            self._is_node = fn._is_node  # type: ignore
        if hasattr(fn, "_node_fn"):
            self._node_fn = fn._node_fn  # type: ignore
        if hasattr(fn, "_service_bindings"):
            self._service_bindings = fn._service_bindings  # type: ignore

    def go(self, *args: Any, **kwargs: Any) -> ObjectRef:
        """Submit this function as a task to the scheduler.

        Returns an ObjectRef immediately without blocking. The function
        will be executed when its dependencies are ready.

        Args:
            *args: Positional arguments (may contain ObjectRefs)
            **kwargs: Keyword arguments (may contain ObjectRefs)

        Returns:
            ObjectRef that will contain the result when execution completes

        Notes:
            If no runtime context is active, a default local runtime is used.

        Example:
            ref1 = process.go("hello")
            ref2 = transform.go(ref1)  # Pass ObjectRef as dependency
            result = await ref2  # Block until complete
        """
        runtime = get_current_runtime()
        runtime.ensure_started()

        # Create task
        task = Task(
            task_id=uuid4(),
            fn=self._fn,  # Use underlying function
            args=args,
            kwargs=kwargs,
        )

        # Submit and return ObjectRef
        return runtime.scheduler.submit_task(task)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call bypasses scheduler and executes immediately.

        This allows the same function to work both synchronously (direct call)
        and asynchronously (via .go()).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function execution
        """
        # Execute original function directly (no scheduler)
        return await maybe_await(self._fn, *args, **kwargs)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"RemoteFunction({self.__name__})"


def wrap_as_remote_function(fn: Callable[..., T]) -> RemoteFunction:
    """Wrap a function to add .go() method.

    Args:
        fn: Function to wrap

    Returns:
        RemoteFunction wrapper with .go() method
    """
    return RemoteFunction(fn)
