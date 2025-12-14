"""
Base interface for all executable components.

Everything that transforms Context inherits from Executable:
- Workers (leaf nodes)
- CompiledGraphs (executable graphs)
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


class Executable:
    """
    Base class for anything that can be executed with a Context.

    Executables implement:
    - arun(ctx) -> ctx: asynchronous execution (required)
    - __call__(ctx) -> ctx: synchronous wrapper (convenience)

    Workers and CompiledGraphs are Executables.
    Graph is a builder, not an Executable - use graph.compile() to get one.
    """

    def __init__(self, name: str | None = None):
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
