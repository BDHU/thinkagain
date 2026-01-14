"""OpFunction - wrapper that adds .go() method to @op functions."""

from typing import Any, Callable, TypeVar, cast
from uuid import uuid4

from ..runtime.utils import maybe_await
from ..runtime.object_ref import ObjectRef
from ..runtime import get_current_runtime
from ..runtime.op import Op

__all__ = ["OpFunction", "wrap_as_op_function"]

T = TypeVar("T")


class OpFunction:
    """Wrapper that adds .go() method to functions.

    This allows functions to be called in two ways:
    1. Direct call: await fn(x) - executes immediately (bypasses scheduler)
    2. Remote call: fn.go(x) - submits to scheduler, returns ObjectRef

    Example:
        @op
        async def process(x: str) -> str:
            return x.upper()

        # Direct execution
        result = await process("hello")

        # Remote execution (via scheduler)
        ref = process.go("hello")
        result = await ref
    """

    def __init__(self, fn: Callable):
        """Initialize the remote operation wrapper.

        Args:
            fn: The function to wrap
        """
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "op_function")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = cast(str, getattr(fn, "__module__", "") or "")
        self.__qualname__ = cast(str, getattr(fn, "__qualname__", "") or "")

        # Copy over special attributes from original function
        if hasattr(fn, "_is_op"):
            self._is_op = fn._is_op  # type: ignore[attr-defined]
        if hasattr(fn, "_op_fn"):
            self._op_fn = fn._op_fn  # type: ignore[attr-defined]
        if hasattr(fn, "_service_bindings"):
            self._service_bindings = fn._service_bindings  # type: ignore[attr-defined]

    def go(self, *args: Any, **kwargs: Any) -> ObjectRef:
        """Submit this function as an op to the scheduler.

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

        # Create op
        op = Op(
            op_id=uuid4(),
            fn=self._fn,  # Use underlying function
            args=args,
            kwargs=kwargs,
        )

        # Submit and return ObjectRef
        return runtime.scheduler.submit_op(op)

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
        return f"OpFunction({self.__name__})"


def wrap_as_op_function(fn: Callable[..., T]) -> OpFunction:
    """Wrap a function to add .go() method.

    Args:
        fn: Function to wrap

    Returns:
        OpFunction wrapper with .go() method
    """
    return OpFunction(fn)
