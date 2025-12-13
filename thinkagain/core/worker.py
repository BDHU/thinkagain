"""
Worker - base class for leaf computations.

Workers are the fundamental units of computation in thinkagain.
They transform Context and can be composed with other executables.

A worker is just an Executable that you implement with business logic.
"""

import inspect
import re
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
    Union,
    overload,
)

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

    All workers must implement async arun(). Synchronous execution
    is available via the __call__() wrapper (runs async under the hood).

    Basic example:
        class VectorDBWorker(Worker):
            async def arun(self, ctx: Context) -> Context:
                ctx.documents = await self.search_async(ctx.query)
                ctx.log(f"[{self.name}] Retrieved {len(ctx.documents)} docs")
                return ctx

    Streaming example (for LLM workers that produce incremental output):
        class StreamingLLMWorker(Worker):
            async def astream(self, ctx: Context) -> AsyncIterator[Context]:
                chunks = []
                async for token in self.llm.stream(ctx.query):
                    chunks.append(token)
                    ctx.response = "".join(chunks)
                    yield ctx  # Yield progressive updates
                # Last yield contains the complete result

    Usage:
        # Compose workers into pipelines
        pipeline = vector_db >> reranker >> generator
        result = await pipeline.arun(ctx)  # async (recommended)
        result = pipeline(ctx)             # sync (runs async under the hood)

        # Stream results from workers that support it
        async for ctx_snapshot in graph.stream(ctx):
            print(ctx_snapshot.response)  # See incremental updates
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

    async def astream(self, ctx: Context) -> AsyncIterator[Context]:
        """
        Stream context updates during execution.

        This is the primary method for workers that produce incremental output
        (e.g., LLM workers streaming tokens). Override this to enable streaming.

        For non-streaming workers, the default implementation calls arun() and
        yields the result once, making all workers compatible with streaming.

        Args:
            ctx: Input context

        Yields:
            Context snapshots with progressive updates. The last yield should
            contain the complete result.

        Example:
            class StreamingLLM(Worker):
                async def astream(self, ctx: Context):
                    chunks = []
                    async for token in self.llm.stream(ctx.query):
                        chunks.append(token)
                        ctx.response = "".join(chunks)
                        yield ctx  # Incremental update
                    # Final yield has complete response
        """
        # Default: call arun and yield once (non-streaming behavior)
        result = await self.arun(ctx)
        yield result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


_AsyncWorkerFunc = TypeVar(
    "_AsyncWorkerFunc", bound=Callable[[Context], Awaitable[Context]]
)


@overload
def async_worker(func: _AsyncWorkerFunc, *, name: Optional[str] = ...) -> Worker: ...


@overload
def async_worker(
    func: None = ..., *, name: Optional[str] = ...
) -> Callable[[_AsyncWorkerFunc], Worker]: ...


def async_worker(
    func: Optional[_AsyncWorkerFunc] = None, *, name: Optional[str] = None
) -> Union[Callable[[_AsyncWorkerFunc], Worker], Worker]:
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

    def _decorate(f: _AsyncWorkerFunc) -> Worker:
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
