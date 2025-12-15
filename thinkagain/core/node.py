"""Node wrapper for lazy execution."""

from __future__ import annotations

from typing import Callable

from .lazy import LazyContext


class Node:
    """Wraps an async function for lazy execution."""

    __slots__ = ("fn", "name")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "node")

    async def __call__(self, ctx: LazyContext | dict) -> LazyContext:
        if isinstance(ctx, dict):
            ctx = LazyContext(ctx)
        return ctx._chain(self)
