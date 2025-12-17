"""Node wrapper for lazy execution."""

from __future__ import annotations

import inspect
from typing import Callable

from .context import Context
from .errors import NodeSignatureError


class Node:
    """Wraps an async function for lazy execution. Works as decorator on functions and methods."""

    __slots__ = ("fn", "name", "_qualname", "_is_method", "_class_name", "_is_bound")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "node")
        self._qualname = getattr(fn, "__qualname__", self.name)
        self._is_method = False
        self._class_name: str | None = None
        self._is_bound = inspect.ismethod(fn)
        # Fallback: infer method metadata from qualname if descriptor hooks are not triggered
        parts = self._qualname.split(".")
        if len(parts) >= 2 and not parts[-2].startswith("<"):
            self._is_method = True
            self._class_name = parts[-2]

    def __set_name__(self, owner, name):
        """Capture class metadata when used as descriptor on a class."""
        if not self.name:
            self.name = name
        self._is_method = True
        self._class_name = getattr(owner, "__name__", None)
        if self._class_name and name:
            self._qualname = f"{self._class_name}.{name}"

    @property
    def is_method(self) -> bool:
        return self._is_method

    @property
    def class_name(self) -> str | None:
        return self._class_name

    @property
    def is_bound(self) -> bool:
        return self._is_bound

    def __get__(self, obj, objtype=None):
        """Descriptor protocol: bind self for methods."""
        if obj is None:
            return self
        bound = Node(self.fn.__get__(obj, objtype), self.name)
        bound._is_method = True
        bound._class_name = self._class_name or getattr(objtype, "__name__", None) or obj.__class__.__name__
        bound._is_bound = True
        return bound

    def __call__(self, ctx: Context | dict | None = None) -> Context:
        """Chain this node to a context."""
        if ctx is None:
            ctx = Context()
        elif isinstance(ctx, dict):
            ctx = Context(ctx)
        return ctx._chain(self)

    async def execute(self, ctx: Context) -> Context:
        """Execute this node."""
        try:
            return await self.fn(ctx)
        except TypeError as e:
            if "argument" in str(e):
                raise NodeSignatureError(self.name, e) from e
            raise


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
