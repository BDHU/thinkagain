"""Shared test fixtures."""

from contextlib import contextmanager
import asyncio

import pytest

from thinkagain import node
from thinkagain.distributed import get_default_manager, reset_backend
from thinkagain.distributed.profiling import disable_profiling


def run_async(coro) -> None:
    """Run a coroutine immediately or schedule it if a loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        loop.create_task(coro)


def _reset_state() -> None:
    disable_profiling()
    get_default_manager().clear()
    reset_backend()


# Shared node definitions - now pure functions
@node
async def add_one(x: int) -> int:
    return x + 1


@node
async def double(x: int) -> int:
    return x * 2


@node
async def append_x(logs: list) -> list:
    return logs + ["x"]


@pytest.fixture
def nodes():
    """Return commonly used nodes."""
    return {"add_one": add_one, "double": double, "append_x": append_x}


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure distributed tests start with a clean registry/backend."""
    _reset_state()
    yield
    _reset_state()


@pytest.fixture
def shutdown_on_exit():
    """Context manager to ensure replicas are shutdown (async-aware)."""

    @contextmanager
    def _shutdown(*replicas):
        try:
            yield
        finally:
            # Shutdown all replicas - need to handle async
            async def _shutdown_all():
                for replica_cls in replicas:
                    await replica_cls.shutdown()

            run_async(_shutdown_all())

    return _shutdown
