"""Shared test fixtures."""

import asyncio

import pytest

from thinkagain import disable_profiling, node


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
    """Ensure distributed tests start with clean state."""
    _reset_state()
    yield
    _reset_state()
