"""Shared test fixtures."""

import pytest

from thinkagain import node


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
