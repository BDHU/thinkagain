"""Node wrapper for lazy execution."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from .context import Context

from .errors import NodeDataclassError


class _ParentRef(NamedTuple):
    """Sentinel marking a parent Context position in call_args."""

    index: int


def _unwrap_args(args: tuple, kwargs: dict) -> tuple[list, dict]:
    """Unwrap Context args to their data values."""
    from .context import Context

    return (
        [a._data if isinstance(a, Context) else a for a in args],
        {k: v._data if isinstance(v, Context) else v for k, v in kwargs.items()},
    )


async def _execute_unwrapped(fn: Callable, args: tuple, kwargs: dict) -> "Context":
    """Execute an async callable with unwrapped args/kwargs."""
    from .context import Context

    exec_args, exec_kwargs = _unwrap_args(args, kwargs)
    return Context(await fn(*exec_args, **exec_kwargs))


class NodeBase(ABC):
    """Abstract base for all node types."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the node name."""
        ...

    def __call__(self, *args, **kwargs) -> "Context":
        """Chain this node with given inputs.

        Context arguments are tracked as parents in the computation graph.
        Other arguments are passed through to the node at execution time.
        """
        from .context import Context

        parents: list[Context] = []
        call_args: list = []
        call_kwargs: dict = {}

        def record_parent(value: Any):
            if isinstance(value, Context):
                parents.append(value)
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

    Subclass this to create stateful nodes with configuration. The run()
    method receives plain Python values (not Context) and returns a plain
    Python value that gets wrapped in Context automatically.

    Example:
        class AddValue(Node):
            delta: int = 1

            async def run(self, x: int) -> int:
                return x + self.delta

        # Instantiate with config
        add_five = AddValue(delta=5)

        # Use in pipeline
        ctx = add_five(Context(10))  # Context(15) after materialization
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
    async def run(self, *args, **kwargs) -> Any:
        """Execute the node logic. Override this in subclasses.

        Receives plain Python values extracted from input Contexts.
        Return a plain Python value (will be wrapped in Context).

        Example:
            async def run(self, x: int) -> int:
                return x + self.delta
        """
        ...

    async def execute(self, *args, **kwargs) -> "Context":
        """Execute this node by calling run() with unwrapped values."""
        return await _execute_unwrapped(self.run, args, kwargs)

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

    The function receives plain Python values (not Context) and returns
    a plain Python value that gets wrapped in Context automatically.

    Example:
        @node
        async def add_one(x: int) -> int:
            return x + 1

        ctx = add_one(Context(5))  # Context(6) after materialization
    """

    __slots__ = ("fn", "_name")

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self._name = name or getattr(fn, "__name__", "node")

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, *args, **kwargs) -> "Context":
        """Execute this node with unwrapped values."""
        return await _execute_unwrapped(self.fn, args, kwargs)


def node(_fn: Callable | None = None, *, name: str | None = None):
    """Decorator that wraps an async function in a FunctionNode.

    The decorated function receives plain Python values and returns
    a plain Python value. The framework handles Context wrapping.

    Example:
        @node
        async def add_one(x: int) -> int:
            return x + 1

        @node
        async def combine(a: int, b: int) -> int:
            return a + b

        # Usage
        ctx = add_one(Context(5))         # Single input
        ctx = combine(Context(1), Context(2))  # Multi-input
    """

    if _fn is None:
        # Used as @node(name="custom")
        def wrapper(fn: Callable) -> FunctionNode:
            return FunctionNode(fn, name=name)

        return wrapper

    # Used as simple @node
    return FunctionNode(_fn, name=name)
