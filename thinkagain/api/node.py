"""@node decorator for marking functions as remotely executable."""

from __future__ import annotations

import functools
import inspect
from typing import Callable, TypeVar, cast

T = TypeVar("T")


def node(
    fn: Callable[..., T] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to mark an async function as a remotely executable node.

    The decorated function can be called in two ways:
    1. Direct call: await fn(x) - executes immediately
    2. Remote call: fn.go(x) - submits to scheduler, returns ObjectRef

    Args:
        fn: The function to decorate (when used without parentheses)
        name: Optional human-readable name for the node (defaults to function name)
        description: Optional description of what the node does

    Example:
        @node
        async def process(text: str) -> str:
            return text.upper()

        @node(name="Text Processor", description="Converts text to uppercase")
        async def process(text: str) -> str:
            return text.upper()

        # Direct execution
        result = await process("hello")

        # Remote execution (dynamic DAG)
        ref = process.go("hello")
        result = await ref
    """
    from .remote_function import RemoteFunction

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        # Only accept async functions
        if not inspect.iscoroutinefunction(fn):
            fn_name = getattr(fn, "__name__", fn.__class__.__name__)
            raise TypeError(
                f"@node requires an async function, got {fn_name}. "
                f"For stateful objects, use @replica instead."
            )

        # Simple wrapper that just executes the function
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        setattr(wrapper, "_is_node", True)
        setattr(wrapper, "_node_fn", fn)
        setattr(wrapper, "_node_name", name or fn.__name__)
        setattr(wrapper, "_node_description", description)

        # Wrap with RemoteFunction to add .go() method
        remote_fn = RemoteFunction(wrapper)
        return cast(Callable[..., T], remote_fn)

    # Support both @node and @node(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator
