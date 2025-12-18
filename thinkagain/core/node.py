"""Node wrapper for lazy execution."""

from __future__ import annotations

import inspect
from typing import Callable

from .context import Context
from .errors import NodeSignatureError


class Node:
    """Wraps an async function for lazy execution.

    Nodes are always stateless functions: async def node(ctx) -> ctx.
    """

    __slots__ = ("fn", "name", "_validated")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "node")
        self._validated = False

    def _validate_signature(self) -> None:
        """Validate signature once, cache result."""
        if self._validated:
            return
        self._validated = True

        try:
            sig = inspect.signature(self.fn)
        except (ValueError, TypeError):
            return  # Can't inspect (builtin, etc.) - allow runtime to catch

        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(required) != 1:
            raise NodeSignatureError(
                self.name,
                TypeError(f"expected 1 required parameter (ctx), got {len(required)}"),
            )

    def __call__(self, ctx: Context | dict | None = None) -> Context:
        """Chain this node to a context."""
        if ctx is None:
            ctx = Context()
        elif isinstance(ctx, dict):
            ctx = Context(ctx)
        return ctx._chain(self)

    async def execute(self, ctx: Context) -> Context:
        """Execute this node."""
        self._validate_signature()
        try:
            return await self.fn(ctx)
        except TypeError as e:
            # Fallback for cases inspect couldn't catch
            if "argument" in str(e) or "positional" in str(e):
                raise NodeSignatureError(self.name, e) from e
            raise


def node(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator that wraps an async function in a Node.

    Nodes are always stateless functions that take a Context and return a Context.

    Supports both @node and @node(name="custom") styles.

    Example:
        @node
        async def process(ctx):
            value = ctx.get("input")
            ctx.set("output", transform(value))
            return ctx
    """

    if _fn is None:
        # Used as @node(name="custom")
        def wrapper(fn: Callable) -> Node:
            return Node(fn, name=name)

        return wrapper

    # Used as simple @node
    return Node(_fn, name=name)
