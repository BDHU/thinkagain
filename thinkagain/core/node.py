"""Node wrapper for lazy execution."""

from __future__ import annotations

from typing import Callable

from .context import Context


class Node:
    """Wraps an async function for lazy execution. Works as decorator on functions and methods."""

    __slots__ = ("fn", "name", "_args", "_kwargs")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "node")
        self._args = ()
        self._kwargs = {}

    def __get__(self, obj, objtype=None):
        """Descriptor protocol: bind self for methods."""
        if obj is None:
            return self
        return Node(self.fn.__get__(obj, objtype), self.name)

    def __call__(self, ctx: Context | dict | None = None, *args, **kwargs) -> Context:
        """Chain this node. Extra args are bound to the function."""
        if ctx is None:
            ctx = Context()
        elif isinstance(ctx, dict):
            ctx = Context(ctx)

        # If extra args/kwargs, create a bound version
        if args or kwargs:
            bound = Node(self.fn, self.name)
            bound._args = args
            bound._kwargs = kwargs
            return ctx._chain(bound)

        return ctx._chain(self)

    async def execute(self, ctx: Context) -> Context:
        """Execute this node with stored args."""
        return await self.fn(ctx, *self._args, **self._kwargs)


def node(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator that wraps an async function in a Node.

    Supports both @node and @node(name="custom") styles.
    """

    if _fn is None:
        # Used as @node(name="custom")
        def wrapper(fn: Callable) -> Node:
            return Node(fn, name=name)

        return wrapper

    # Used as simple @node
    return Node(_fn, name=name)
