"""Runtime helpers shared by tracing and execution."""

from __future__ import annotations

import inspect
from typing import Any, Callable


async def maybe_await(fn: Callable, *args, **kwargs) -> Any:
    """Call a function and await if it returns an awaitable."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
