"""Shared test fixtures."""

import pytest

from thinkagain import node


# Shared node definitions
@node
async def add_one(ctx):
    ctx.set("value", ctx.get("value", 0) + 1)
    return ctx


@node
async def double(ctx):
    ctx.set("value", ctx.get("value") * 2)
    return ctx


@node
async def append_x(ctx):
    ctx.set("logs", ctx.get("logs", []) + ["x"])
    return ctx


@pytest.fixture
def nodes():
    """Return commonly used nodes."""
    return {"add_one": add_one, "double": double, "append_x": append_x}
