"""Replicate decorator for distributed execution."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

from ...core.tracing import node


@dataclass
class DistributionConfig:
    """Configuration for distributed execution of a function."""

    gpus: int | None = None
    backend: str = "local"
    setup: Callable | None = None


class ReplicatedCallable:
    """Wrapper that makes a @replicate decorated function/class serveable.

    This wrapper serves two purposes:
    1. When called normally (in @jit pipeline), it executes the function
    2. When passed to thinkagain.serve, it can be instantiated as a server
    """

    def __init__(self, fn: Callable, config: DistributionConfig):
        """Initialize replicated callable.

        Args:
            fn: The original function or class
            config: Distribution configuration
        """
        self._fn = fn
        self._distribution_config = config

        # Preserve original metadata
        self.__name__ = getattr(fn, "__name__", repr(fn))
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__module__ = getattr(fn, "__module__", None)
        self.__doc__ = fn.__doc__
        self.__annotations__ = getattr(fn, "__annotations__", {})

    def __call__(self, *args, **kwargs):
        """Call the underlying function/class."""
        return self._fn(*args, **kwargs)

    def __get__(self, obj, objtype=None):
        """Support for method binding."""
        if obj is None:
            return self
        return lambda *args, **kwargs: self._fn(obj, *args, **kwargs)


def replicate(
    gpus: int | None | Callable = None,
    backend: str = "local",
    setup: Callable | None = None,
):
    """Decorator for replicated execution across multiple instances.

    Marks a function or class for distributed execution. The decorated object
    can be:
    1. Called from within a @jit function (routes to mesh instances)
    2. Served directly with: python -m thinkagain.serve module:function

    Can be used with or without parentheses:
        @replicate
        @replicate()
        @replicate(gpus=1)

    Args:
        gpus: GPUs required per instance (None = CPU-only, int = GPU count)
        backend: Execution backend ("local" or "grpc")
        setup: Optional function to initialize state per instance.
               If provided, the setup function is called once per instance,
               and its return value is passed as the first argument to the
               replicated function.

    Example:
        # Simple function replication
        @replicate(backend="grpc")
        async def reverse_text(text: str) -> str:
            return text[::-1]

        # Serve it:
        # python -m thinkagain.serve my_module:reverse_text --port 8000

        # Or use in pipeline:
        mesh = Mesh([MeshNode("local", endpoint="localhost:8000")])

        @jit
        async def pipeline(text: str) -> str:
            return await reverse_text(text)

        with mesh:
            result = await pipeline("hello")  # -> "olleh"

        # Class-based replication with state
        @replicate(gpus=1, backend="grpc")
        class LLMServer:
            def __init__(self):
                self.model = load_model()

            async def __call__(self, prompt: str) -> str:
                return self.model.generate(prompt)

        # Serve it:
        # python -m thinkagain.serve my_module:LLMServer --port 8000
    """

    def decorator(fn):
        config = DistributionConfig(
            gpus=_gpus,
            backend=backend,
            setup=setup,
        )

        # Apply @node decorator to async functions and classes with async __call__
        # so they integrate with @jit. The distributed_execution_hook will intercept
        # during graph execution.
        is_async_fn = inspect.iscoroutinefunction(fn)
        is_async_callable = inspect.iscoroutinefunction(getattr(fn, "__call__", None))

        if is_async_fn or is_async_callable:
            # Apply @node to async functions and classes with async __call__
            wrapped = node(fn)
            # Preserve distribution config on the node-wrapped version
            wrapped._distribution_config = config
            wrapped._fn = fn
        else:
            # For sync functions, wrap in ReplicatedCallable
            wrapped = ReplicatedCallable(fn, config)

        return wrapped

    # Handle @replicate without parentheses
    if callable(gpus):
        fn = gpus
        _gpus = None
        return decorator(fn)

    # Handle @replicate() with parentheses
    _gpus = gpus
    return decorator
