"""
Base class for all executable components.

Executables transform Context and can be composed into graphs.
Subclass this to implement your business logic.
"""

import asyncio
import inspect
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


class Executable:
    """
    Base class for components that transform Context.

    Subclasses implement arun() for the core logic.
    Override astream() for streaming output (e.g., LLM token streaming).

    Example:
        class MyWorker(Executable):
            async def arun(self, ctx: Context) -> Context:
                ctx.result = await process(ctx.input)
                return ctx

        class StreamingLLM(Executable):
            async def astream(self, ctx: Context) -> AsyncIterator[Context]:
                chunks = []
                async for token in self.llm.stream(ctx.query):
                    chunks.append(token)
                    ctx.response = "".join(chunks)
                    yield ctx
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name if name is not None else self.__class__.__name__.lower()

    def __call__(self, ctx: Context) -> Context:
        """Synchronous execution wrapper."""
        return asyncio.run(self.arun(ctx))

    async def arun(self, ctx: Context) -> Context:
        """Execute and return modified context. Subclasses must implement."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement arun()")

    async def astream(self, ctx: Context) -> AsyncIterator[Context]:
        """
        Stream context updates. Override for incremental output.

        Default: calls arun() and yields once.
        """
        result = await self.arun(ctx)
        yield result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# Type for async_executable decorator
_AsyncFunc = TypeVar("_AsyncFunc", bound=Callable[[Context], Awaitable[Context]])


@overload
def async_executable(func: _AsyncFunc, *, name: Optional[str] = ...) -> Executable: ...


@overload
def async_executable(
    func: None = ..., *, name: Optional[str] = ...
) -> Callable[[_AsyncFunc], Executable]: ...


def async_executable(
    func: Optional[_AsyncFunc] = None, *, name: Optional[str] = None
) -> Union[Callable[[_AsyncFunc], Executable], Executable]:
    """
    Decorator to wrap an async function as an Executable.

    Usage:
        @async_executable
        async def fetch(ctx: Context) -> Context:
            ctx.data = await get_data()
            return ctx

        graph.add("fetch", fetch)
    """

    def _decorate(f: _AsyncFunc) -> Executable:
        if not inspect.iscoroutinefunction(f):
            raise TypeError("async_executable expects an async function")

        exec_name = name or f.__name__

        class _AsyncFuncExecutable(Executable):
            async def arun(self, ctx: Context) -> Context:
                return await f(ctx)

            def __repr__(self) -> str:
                return f"AsyncFuncExecutable(name='{self.name}', func='{f.__name__}')"

        return _AsyncFuncExecutable(exec_name)

    if func is None:
        return _decorate
    return _decorate(func)
