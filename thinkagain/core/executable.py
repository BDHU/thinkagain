"""
Base interface for all executable components.

Everything that transforms Context inherits from Executable:
- Workers (leaf nodes)
- Graphs (DAGs with cycles)
- Functions (raw callables)
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context
    from .graph import Graph


class Executable:
    """
    Base class for anything that transforms Context.

    All components in thinkagain implement this interface:
    - arun(ctx) -> ctx: asynchronous execution (required)
    - __call__(ctx) -> ctx: synchronous wrapper (provided automatically)
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

    def __call__(self, ctx: "Context") -> "Context":
        """
        Execute synchronously (convenience wrapper).

        This runs the async arun() method using asyncio.run().

        Args:
            ctx: Input context

        Returns:
            Modified context
        """
        return asyncio.run(self.arun(ctx))

    async def arun(self, ctx: "Context") -> "Context":
        """
        Execute asynchronously.

        Args:
            ctx: Input context

        Returns:
            Modified context

        Note:
            Subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement arun()")

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
