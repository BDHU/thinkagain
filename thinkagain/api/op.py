"""@op decorator for marking functions as remotely executable operations."""

from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar, cast

T = TypeVar("T")


def op(
    fn: Callable[..., T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to mark an async function as a remotely executable operation.

    The decorated function can be called in two ways:
    1. Direct call: await fn(x) - executes immediately
    2. Remote call: fn.go(x) - submits to scheduler, returns ObjectRef

    Args:
        fn: The function to decorate (when used without parentheses)
        name: Optional human-readable name for the operation (defaults to function name)
        description: Optional description of what the operation does

    Example:
        @op
        async def process(text: str) -> str:
            return text.upper()

        @op(name="Text Processor", description="Converts text to uppercase")
        async def process(text: str) -> str:
            return text.upper()

        # Direct execution
        result = await process("hello")

        # Remote execution (dynamic DAG)
        ref = process.go("hello")
        result = await ref
    """
    from .op_function import OpFunction

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        # Only accept async functions
        if not inspect.iscoroutinefunction(fn):
            fn_name = getattr(fn, "__name__", fn.__class__.__name__)
            raise TypeError(
                f"@op requires an async function, got {fn_name}. "
                f"For stateful objects, use @service instead."
            )

        # Simple wrapper that just executes the function
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        setattr(wrapper, "_is_op", True)
        setattr(wrapper, "_op_fn", fn)
        fn_name = getattr(fn, "__name__", "<anonymous>")
        setattr(wrapper, "_op_name", name or fn_name)
        setattr(wrapper, "_op_description", description)

        # Wrap with OpFunction to add .go() method
        op_function = OpFunction(wrapper)
        return cast(Callable[..., T], op_function)

    # Support both @op and @op(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator
