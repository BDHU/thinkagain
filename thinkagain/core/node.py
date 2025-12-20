"""Node wrapper for lazy execution."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .context import Context

from .errors import NodeDataclassError, NodeSignatureError


class NodeBase(ABC):
    """Abstract base for all node types."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the node name."""
        ...

    def __call__(self, ctx: "Context | dict | None" = None) -> "Context":
        """Chain this node to a context."""
        from .context import Context

        if ctx is None:
            ctx = Context()
        elif isinstance(ctx, dict):
            ctx = Context(ctx)
        return ctx._chain(self)

    @abstractmethod
    async def execute(self, ctx: "Context") -> "Context":
        """Execute this node."""
        ...


class Node(NodeBase):
    """Base class for dataclass-based nodes.

    Subclass this to create stateful nodes with configuration that can be
    serialized. Subclasses are automatically made frozen dataclasses.

    IMPORTANT: Do NOT apply @dataclass yourself - it is applied automatically.

    Example:
        class Summarize(Node):
            model: str = "gpt-4"
            max_tokens: int = 100

            async def run(self, ctx: Context) -> Context:
                # Use self.model, self.max_tokens
                return ctx.set("summary", result)

        # Instantiate with config
        summarizer = Summarize(model="claude", max_tokens=200)

        # Use in pipeline
        ctx = summarizer(ctx)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclasses are frozen dataclasses and validate run() signature."""
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        if is_dataclass(cls):
            params = getattr(cls, "__dataclass_params__", None)
            if not params or not params.frozen:
                raise NodeDataclassError(
                    cls.__name__,
                    "Dataclass must be frozen=True.",
                )
        else:
            if "__init__" in cls.__dict__:
                raise NodeDataclassError(
                    cls.__name__,
                    "Node subclasses auto-apply @dataclass(frozen=True); "
                    "define fields as class attributes and use __post_init__ if needed.",
                )

            dataclass(frozen=True)(cls)

        # Validate run() signature at definition time
        if "run" in cls.__dict__:
            try:
                sig = inspect.signature(cls.run)
            except (ValueError, TypeError):
                return  # Can't inspect, skip validation

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
            # Expect 2 params: self and ctx
            if len(required) != 2:
                raise NodeSignatureError(
                    cls.__name__,
                    TypeError(
                        f"expected 1 required parameter (ctx), got {len(required) - 1}"
                    ),
                )

    @property
    def name(self) -> str:
        """Return the class name as the node name."""
        return self.__class__.__name__

    @abstractmethod
    async def run(self, ctx: "Context") -> "Context":
        """Execute the node logic. Override this in subclasses."""
        ...

    async def execute(self, ctx: "Context") -> "Context":
        """Execute this node by calling run()."""
        return await self.run(ctx)

    def serialize(self) -> dict[str, Any]:
        """Serialize this node for distribution."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "qualname": self.__class__.__qualname__,
            "config": {field.name: getattr(self, field.name) for field in fields(self)},
        }


class FunctionNode(NodeBase):
    """Wraps an async function for lazy execution.

    This is the original Node implementation, renamed to FunctionNode.
    Nodes are always stateless functions: async def node(ctx) -> ctx.

    Signature validation happens eagerly at decoration time for consistent
    behavior with class-based Node validation.
    """

    __slots__ = ("fn", "_name")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self._name = name or getattr(fn, "__name__", "node")
        # Eager validation at construction time (fail fast)
        self._validate_signature()

    @property
    def name(self) -> str:
        return self._name

    def _validate_signature(self) -> None:
        """Validate signature (called once at construction)."""
        try:
            sig = inspect.signature(self.fn)
        except (ValueError, TypeError):
            # Can't inspect (builtin, etc.) - defer to runtime
            return

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
                self._name,
                TypeError(f"expected 1 required parameter (ctx), got {len(required)}"),
            )

    async def execute(self, ctx: "Context") -> "Context":
        """Execute this node."""
        try:
            return await self.fn(ctx)
        except TypeError as e:
            # Fallback for cases inspect couldn't catch
            if "argument" in str(e) or "positional" in str(e):
                raise NodeSignatureError(self._name, e) from e
            raise


def node(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator that wraps an async function in a FunctionNode.

    Nodes are always stateless functions that take a Context and return a Context.

    Supports both @node and @node(name="custom") styles.

    Example:
        @node
        async def process(ctx):
            value = ctx.get("input")
            ctx.set("output", transform(value))
            return ctx

    For stateful nodes with configuration, use the Node base class instead:

        class Process(Node):
            config: str = "default"

            async def run(self, ctx):
                return ctx.set("output", self.config)
    """

    if _fn is None:
        # Used as @node(name="custom")
        def wrapper(fn: Callable) -> FunctionNode:
            return FunctionNode(fn, name=name)

        return wrapper

    # Used as simple @node
    return FunctionNode(_fn, name=name)
