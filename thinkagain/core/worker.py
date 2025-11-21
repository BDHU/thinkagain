"""
Worker - base class for leaf computations.

Workers are the fundamental units of computation in thinkagain.
They transform Context and can be composed with other executables.

A worker is just an Executable that you implement with business logic.
"""

import re
import inspect
from .context import Context
from .executable import Executable


class Worker(Executable):
    """
    Base class for workers in the agent framework.

    Workers are units of computation that:
    - Take a Context as input
    - Perform some operation
    - Return the modified Context
    - Log their actions for debugging
    - Can be composed using >> operator

    Workers can be simple functions, complex stateful services,
    or anything in between (e.g., vector DB, LLM, reranker).

    Supports both synchronous and asynchronous execution:
    - Implement __call__(self, ctx) for sync workers
    - Implement async arun(self, ctx) for async workers
    - Executable handles the appropriate bridging when only one of them
      is provided

    Synchronous example:
        class VectorDBWorker(Worker):
            def __call__(self, ctx: Context) -> Context:
                ctx.documents = self.search(ctx.query)
                ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} docs")
                return ctx

    Asynchronous example:
        class AsyncVectorDBWorker(Worker):
            async def arun(self, ctx: Context) -> Context:
                ctx.documents = await self.search_async(ctx.query)
                ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} docs")
                return ctx

    Dual support example:
        class DualVectorDBWorker(Worker):
            def __call__(self, ctx: Context) -> Context:
                ctx.documents = self.search(ctx.query)
                return ctx

            async def arun(self, ctx: Context) -> Context:
                ctx.documents = await self.search_async(ctx.query)
                return ctx

    Usage:
        # Compose workers into pipelines
        pipeline = vector_db >> reranker >> generator
        result = await pipeline.arun(ctx)  # async
        result = pipeline(ctx)             # sync
    """

    def __init__(self, name: str = None):
        """
        Initialize worker with a name.

        Args:
            name: Identifier for this worker (used in logging).
                  If not provided, auto-generates from class name.
        """
        # Generate name from class if not provided
        if name is None:
            name = self._generate_name()

        super().__init__(name)

    def _generate_name(self) -> str:
        """
        Generate a name from the class name.

        Examples:
            VectorDBWorker -> vector_db_worker
            RerankerWorker -> reranker_worker
        """
        class_name = self.__class__.__name__
        # Convert CamelCase to snake_case
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name

    def __call__(self, ctx: Context) -> Context:
        """
        Process the context and return modified context (synchronous).

        Args:
            ctx: Input context

        Returns:
            Modified context

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError(f"Worker '{self.name}' must implement __call__")

    def to_dict(self) -> dict:
        """Export worker structure as dictionary."""
        return {"type": "Worker", "name": self.name, "class": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def async_worker(func=None, *, name: str = None):
    """
    Decorator that wraps an async function into a Worker instance.

    Usage:
        @async_worker
        async def fetch(ctx: Context) -> Context:
            ...
            return ctx

        pipeline = fetch >> another_worker

    Args:
        func: The async function with signature ``(ctx: Context) -> Context``.
        name: Optional worker name; defaults to ``func.__name__``.
    """

    def _decorate(f):
        if not inspect.iscoroutinefunction(f):
            raise TypeError("async_worker expects an async function")

        worker_name = name or f.__name__

        class _AsyncFuncWorker(Worker):
            async def arun(self, ctx: Context) -> Context:
                return await f(ctx)

            def __repr__(self) -> str:
                return f"AsyncFuncWorker(name='{self.name}', func='{f.__name__}')"

        return _AsyncFuncWorker(worker_name)

    if func is None:
        return _decorate

    return _decorate(func)
