"""
Base interface for all executable components.

Everything that transforms Context inherits from Executable:
- Workers (leaf nodes)
- Graphs (DAGs with cycles)
- Functions (raw callables)
"""

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


async def run_sync(func, *args, **kwargs):
    """Run blocking ``func`` in a background thread and await the result."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _target():
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - thread path
            loop.call_soon_threadsafe(future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(future.set_result, result)

    threading.Thread(target=_target, daemon=True).start()
    return await future


class Executable:
    """
    Base class for anything that transforms Context.

    All components in thinkagain implement this interface:
    - __call__(ctx) -> ctx: synchronous execution
    - arun(ctx) -> ctx: asynchronous execution
    - __rshift__(other) -> Graph: composition via >> operator

    This unified interface enables seamless composition:
        worker1 >> worker2 >> graph1 >> worker3

    Everything is composable with everything.
    """

    def __init__(self, name: str = None):
        """
        Initialize executable with a name.

        Args:
            name: Identifier for this executable (used in logging)
        """
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        """Generate default name from class name."""
        return self.__class__.__name__.lower()

    @staticmethod
    def _overrides(instance: "Executable", method_name: str) -> bool:
        """Return True when ``method_name`` is implemented by a subclass."""
        base_impl = getattr(Executable, method_name, None)
        current_impl = getattr(type(instance), method_name, None)
        return current_impl is not None and current_impl is not base_impl

    def __call__(self, ctx: "Context") -> "Context":
        """Execute synchronously, falling back to async implementation if needed."""
        if self._overrides(self, "arun"):
            return asyncio.run(self.arun(ctx))
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")

    async def arun(self, ctx: "Context") -> "Context":
        """
        Execute asynchronously.

        Args:
            ctx: Input context

        Returns:
            Modified context

        Note:
            Default implementation wraps __call__() for sync components
        """
        if self._overrides(self, "__call__"):
            return await run_sync(self.__call__, ctx)

        raise NotImplementedError(f"{self.__class__.__name__} must implement arun")

    def __rshift__(self, other) -> "Graph":
        """
        Compose executables using >> operator.

        This always treats both operands as black boxes, creating a simple
        2-node sequential graph. This ensures associativity:
            (A >> B) >> C  ≡  A >> (B >> C)

        To flatten subgraphs, use compile():
            pipeline = worker1 >> subgraph >> worker2
            flat = pipeline.compile()

        Args:
            other: Next executable in the chain

        Returns:
            Graph containing both executables in sequence

        Example:
            # Black-box composition (always)
            pipeline = worker1 >> worker2 >> graph1 >> worker3

            # Execute with nested subgraphs
            result = await pipeline.arun(ctx)

            # Or compile flat for debugging
            flat = pipeline.compile()
            print(flat.visualize())
        """
        from .graph import Graph, END

        # ALWAYS create simple sequential graph (black box)
        # No special handling for Graph types - treat everything uniformly
        g = Graph(name=f"{self.name}_seq")
        g.add_node("_0", self)
        g.add_node("_1", other)
        g.set_entry("_0")
        g.add_edge("_0", "_1")
        g.add_edge("_1", END)
        return g

    def to_dict(self) -> dict:
        """Export structure as dictionary for inspection/serialization."""
        return {"type": self.__class__.__name__, "name": self.name}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
