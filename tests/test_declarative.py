"""Tests for declarative API."""

import asyncio
import pytest
from thinkagain import Node, run, Context, LazyContext, NeedsMaterializationError


# Test functions
async def add_one(ctx):
    ctx.value = ctx.get("value", 0) + 1
    return ctx


async def double(ctx):
    ctx.value = ctx.value * 2
    return ctx


async def set_score(ctx, score):
    ctx.score = score
    return ctx


async def append_log(ctx, msg):
    ctx.logs = ctx.get("logs", []) + [msg]
    return ctx


# Nodes
add = Node(add_one)
dbl = Node(double)


def test_linear_pipeline():
    async def pipeline(ctx):
        ctx = await add(ctx)
        ctx = await dbl(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))
    assert result.value == 12  # (5+1)*2


def test_empty_pipeline():
    async def pipeline(ctx):
        return ctx

    result = asyncio.run(run(pipeline, {"value": 42}))
    assert result.value == 42


def test_dict_input():
    async def pipeline(ctx):
        ctx = await add(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 10}))
    assert result.value == 11


def test_none_input():
    async def pipeline(ctx):
        ctx = await add(ctx)
        return ctx

    result = asyncio.run(run(pipeline))
    assert result.value == 1


def test_explicit_materialization():
    async def pipeline(ctx):
        ctx = await add(ctx)
        ctx = await ctx
        assert ctx.value == 6
        ctx = await dbl(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))
    assert result.value == 12


def test_access_before_materialization_raises():
    async def pipeline(ctx):
        ctx = await add(ctx)
        _ = ctx.value
        return ctx

    with pytest.raises(NeedsMaterializationError):
        asyncio.run(run(pipeline, {"value": 5}))


def test_set_before_materialization_raises():
    async def pipeline(ctx):
        ctx = await add(ctx)
        ctx.foo = "bar"
        return ctx

    with pytest.raises(NeedsMaterializationError):
        asyncio.run(run(pipeline, {"value": 5}))


def test_if_else_branching():
    async def high(ctx):
        ctx.score = 0.9
        return ctx

    async def low(ctx):
        ctx.score = 0.3
        return ctx

    high_node = Node(high)
    low_node = Node(low)

    async def pipeline(ctx):
        if ctx.use_high:
            ctx = await high_node(ctx)
        else:
            ctx = await low_node(ctx)
        return ctx

    assert asyncio.run(run(pipeline, {"use_high": True})).score == 0.9
    assert asyncio.run(run(pipeline, {"use_high": False})).score == 0.3


def test_conditional_after_materialization():
    async def log_high(ctx):
        ctx.logs = ["high"]
        return ctx

    async def log_low(ctx):
        ctx.logs = ["low"]
        return ctx

    h = Node(log_high)
    l = Node(log_low)

    async def pipeline(ctx):
        ctx = await add(ctx)
        ctx = await ctx
        if ctx.value > 5:
            ctx = await h(ctx)
        else:
            ctx = await l(ctx)
        return ctx

    assert asyncio.run(run(pipeline, {"value": 10})).logs == ["high"]
    assert asyncio.run(run(pipeline, {"value": 2})).logs == ["low"]


def test_while_loop():
    async def pipeline(ctx):
        ctx = await ctx
        while ctx.value < 5:
            ctx = await add(ctx)
            ctx = await ctx
        return ctx

    result = asyncio.run(run(pipeline, {"value": 0}))
    assert result.value == 5


def test_for_loop():
    async def append_x(ctx):
        ctx.logs = ctx.get("logs", []) + ["x"]
        return ctx

    log = Node(append_x)

    async def pipeline(ctx):
        for _ in range(3):
            ctx = await log(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {}))
    assert result.logs == ["x", "x", "x"]


def test_node_name():
    assert add.name == "add_one"


def test_node_explicit_name():
    n = Node(add_one, name="custom")
    assert n.name == "custom"


def test_lazy_context_is_pending():
    ctx = LazyContext({})
    assert not ctx.is_pending
    ctx = LazyContext({}, [add])
    assert ctx.is_pending


def test_nested_functions():
    async def add_twice(ctx):
        ctx = await add(ctx)
        ctx = await add(ctx)
        return ctx

    async def pipeline(ctx):
        ctx = await add_twice(ctx)
        ctx = await dbl(ctx)
        return ctx

    result = asyncio.run(run(pipeline, {"value": 5}))
    assert result.value == 14  # (5+1+1)*2


def test_early_return():
    async def pipeline(ctx):
        ctx = await add(ctx)
        ctx = await ctx
        if ctx.value > 10:
            return ctx
        ctx = await dbl(ctx)
        return ctx

    assert asyncio.run(run(pipeline, {"value": 5})).value == 12
    assert asyncio.run(run(pipeline, {"value": 15})).value == 16
