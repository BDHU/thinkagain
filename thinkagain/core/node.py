"""Node wrapper for lazy execution."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .context import Context

from .errors import NodeDataclassError


class _ParentRef:
    """Sentinel marking a parent Context position in call_args.

    Each instance corresponds to a Context argument that will be resolved
    at execution time. The index indicates which parent in _parents to use.
    """

    __slots__ = ("index",)

    def __init__(self, index: int):
        self.index = index

    def __repr__(self) -> str:
        return f"ParentRef({self.index})"


class NodeBase(ABC):
    """Abstract base for all node types."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the node name."""
        ...

    def __call__(self, *args, **kwargs) -> "Context":
        """Chain this node with given inputs.

        Context/dict arguments are tracked as parents in the computation graph,
        including those provided via kwargs. Other arguments are passed through
        to the node at execution time.
        """
        from .context import Context

        parents: list[Context] = []
        call_args: list = []
        call_kwargs: dict = {}

        def record_parent(value: Any):
            if isinstance(value, Context):
                parents.append(value)
                return _ParentRef(len(parents) - 1)
            if isinstance(value, dict):
                parents.append(Context(value))
                return _ParentRef(len(parents) - 1)
            return value

        for arg in args:
            call_args.append(record_parent(arg))

        for key, value in kwargs.items():
            call_kwargs[key] = record_parent(value)

        # Ensure at least one parent context
        if not parents:
            parents.append(Context())
            call_args.insert(0, _ParentRef(0))

        return parents[0]._chain(self, tuple(parents), tuple(call_args), call_kwargs)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> "Context":
        """Execute this node with the given arguments."""
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
        """Ensure subclasses are frozen dataclasses."""
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

    @property
    def name(self) -> str:
        """Return the class name as the node name."""
        return self.__class__.__name__

    @abstractmethod
    async def run(self, *args, **kwargs) -> "Context":
        """Execute the node logic. Override this in subclasses.

        Receives materialized Context arguments and any additional arguments
        passed when the node was called. For single-input nodes, typically:

            async def run(self, ctx: Context) -> Context: ...

        For multi-input nodes:

            async def run(self, ctx1: Context, ctx2: Context) -> Context: ...
        """
        ...

    async def execute(self, *args, **kwargs) -> "Context":
        """Execute this node by calling run()."""
        return await self.run(*args, **kwargs)

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

    Nodes can accept any number of arguments. Context arguments are tracked
    in the computation graph and materialized before execution.
    """

    __slots__ = ("fn", "_name")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self._name = name or getattr(fn, "__name__", "node")

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, *args, **kwargs) -> "Context":
        """Execute this node with the given arguments."""
        return await self.fn(*args, **kwargs)


def node(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator that wraps an async function in a FunctionNode.

    Supports both @node and @node(name="custom") styles.

    Example (single input):
        @node
        async def process(ctx):
            value = ctx.get("input")
            ctx.set("output", transform(value))
            return ctx

    Example (multi-input):
        @node
        async def merge(ctx1, ctx2):
            ctx1.set("combined", ctx1.get("a") + ctx2.get("b"))
            return ctx1

        result = merge(branch_a, branch_b)

    For stateful nodes with configuration, use the Node base class instead.
    """

    if _fn is None:
        # Used as @node(name="custom")
        def wrapper(fn: Callable) -> FunctionNode:
            return FunctionNode(fn, name=name)

        return wrapper

    # Used as simple @node
    return FunctionNode(_fn, name=name)
